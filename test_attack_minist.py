import torch
#from wideresnet import *
import os, argparse, logging, sys, shutil
import numpy as np
import utils
#import shutil
import matplotlib.pyplot as plt
from attack import *
from models import PytorchModel
from paper_model import vgg16, BasicCNN
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
#from robustbench.utils import load_model as load_model_aa
from patch.models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from patch.models.resnet import ResNet50, ResNet152, ResNet101
import gol as gl
import math
import utils_sima as utl
import torchvision.datasets as dset
import torchvision.models as models
from  torchvision import utils as vutils

a = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Dataset to be used, [MNIST, CIFAR10, Imagenet]')
parser.add_argument('--attack', type=str, default="OPT_attack", 
                    help='Attack to be used')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack.')
parser.add_argument('--random_start', action='store_true', default=False,
                    help='PGD attack with random start.')
parser.add_argument('--fd_eta', type=float, help='\eta, used to estimate the derivative via finite differences')
parser.add_argument('--image_lr', type=float, help='Learning rate for the image (iterative attack)')
parser.add_argument('--online_lr', type=float, help='Learning rate for the prior')
parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior') 
parser.add_argument('--epsilon', type=float, default=0.01,
                        help='epsilon in the PGD attack')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test batch_size')
parser.add_argument('--test_batch', type=int, default=100,
                    help='test batch number')
parser.add_argument('--model', type=str, default="mnist",required=True, help='model to be attacked')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--query', type=int, default=11000, help='Query limit allowed')
parser.add_argument('--save', type=str, default='zc', help='exp_id')
parser.add_argument('--exp_tag', type=str, default='')
parser.add_argument('--gpu', type=str, default='auto', help='tag for saving, enter debug mode if debug is in it')
parser.add_argument('--data_root', type=str, default='D:\\zc\\L_p-norm_Distortion_Efficient_Adversarial_Attack',required=True, help='root directory of imagenet data')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')

args = parser.parse_args()
#### env
np.random.seed(args.seed)
torch.manual_seed(args.seed)
'''gpu = utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
torch.cuda.set_device(gpu)
print('gpu:', gpu)'''


#### macros
attack_list = {
    "PGD":PGD,
    "Sign_OPT": OPT_attack_sign_SGD,
    "Sign_OPT_lf": OPT_attack_sign_SGD_lf,
    "CW": CW,
    "OPT_attack": OPT_attack,
    "HSJA": HSJA,
    "OPT_attack_lf": OPT_attack_lf,
    "FGSM": FGSM,
    "NES": NES,
    "Bandit": Bandit,
    "NATTACK": NATTACK,
    "Sign_SGD": Sign_SGD,
    "ZOO": ZOO,
    "Liu": OPT_attack_sign_SGD_v2,
    "Evolutionary": Evolutionary,
    "SimBA": SimBA
}

l2_list = ["Sign_OPT","CW", "OPT_attack","FGSM","ZOO","SimBA"]
linf_list = ["PGD","Sign_OPT_lf","OPT_attack_lf","NES", "Sign_OPT_lf_bvls"]

if args.attack in l2_list:
    norm = 'L2'
elif args.attack in linf_list:
    norm = 'Linf'


#### dir managemet
exp_id = args.save
args.save = './experiments/{}-{}'\
    .format(exp_id, args.model)
if args.exp_tag != '':
    args.save += '-{}'.format(args.exp_tag)

scripts_to_save = ['./exp_scripts/{}'.format(exp_id + '.sh')]
if os.path.exists(args.save):
    if 'debug' in args.exp_tag or input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
        shutil.rmtree(args.save)
    else: exit()
utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)


#### logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)
if os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else: exit()
fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


#### load data
if args.dataset == "MNIST":
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
elif args.dataset == 'CIFAR10':
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data(args.test_batch_size)
elif args.dataset == 'Imagenet':
    print('unsupported right now')
    exit(1)
else:
    print("Unsupport dataset")

logging.info(args)

#### load model
## load defense model
## clean
aa_model_dir = './model/defense_models'
if   args.model == 'mnist':
    model = MNIST()
    #model = BasicCNN()
    model = torch.nn.DataParallel(model, device_ids=[0])
    load_model(model, 'model/mnist_gpu.pt')
elif args.model == 'cifar10':
    # model = vgg16()
    # model = VGG_plain('VGG16', 10, img_width=32)
    #model = WideResmodel().to(device)
    model = CIFAR10()
    model = torch.nn.DataParallel(model, device_ids=[0])
    load_model(model, 'model/cifar10_gpu.pt')
## linf
elif args.model == 'Sehwag2020Hydra' or args.model == 'hydra': # Hydra 
    model = load_model_aa(model_name='Sehwag2020Hydra', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Wang2020Improving' or args.model == 'mart': # 
    model = load_model_aa(model_name='Wang2020Improving', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Zhang2019Theoretically' or args.model == 'trades': # TRADES
    model = load_model_aa(model_name='Zhang2019Theoretically', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Wong2020Fast' or args.model == 'fastat': # Fast AT
    model = load_model_aa(model_name='Wong2020Fast', model_dir=aa_model_dir, norm=norm)
## l2yyy
elif args.model == 'Wu2020Adversarial':
    model = load_model_aa(model_name='Wu2020Adversarial', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Augustin2020Adversarial':
    model = load_model_aa(model_name='Augustin2020Adversarial', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Rice2020Overfitting':
    model = load_model_aa(model_name='Rice2020Overfitting', model_dir=aa_model_dir, norm=norm)

elif args.model == 'resnet18':
    model = getattr(models, args.model)(pretrained=True).to(device)

elif args.model == 'resnet50':
    model = getattr(models, args.model)(pretrained=True).to(device)

else:
    print('unsupported model'); exit(1)

model.cuda()
model.eval()

## load attack model
# sign opt
amodel = PytorchModel(model, bounds=[0,1], num_classes=10) # just a wrapper
if args.attack=="Bandit":
    attack = attack_list[args.attack](amodel,args.exploration,args.fd_eta,args.online_lr,args.mode)
else:
    attack = attack_list[args.attack](amodel)

total_r_count = 0
total_clean_count = 0
total_distance = 0
rays_successes = []
successes = []
stop_queries = [] # wrc added to match RayS
Linf_distances_list = []
L2_distances_list = []

#################################  ImageNet Load #####################
'''batchfile = 'save\images1000.pth' #% (args.sampled_image_dir, args.model, args.num_runs)

testset = dset.ImageFolder(args.data_root + '/val', utl.IMAGENET_TRANSFORM)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    images = torch.zeros(args.num_runs, 3, 224, 224)
    labels = torch.zeros(args.num_runs).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utl.get_preds(model, images[idx], 'imagenet', batch_size=args.batch_size)
    torch.save({'images': images, 'labels': labels}, batchfile)'''
###################################   

total_zero_num = 0

gl._init()
gl.set_value('total_zero_num', 0)
gl.set_value('under_thold', 0)


#N = int(math.floor(args.test_batch / float(args.batch_size)))
attack_num = 0
success_num = 0
for i, (xi, yi) in enumerate(test_loader):
    logging.info(f"image batch: {i}")
    
    ## data
    if i == args.test_batch: break
    xi, yi = xi.cuda(), yi.cuda()
    ## attack
    theta_init = None
    if (torch.max(amodel.predict(xi),0)[1] != yi):  
        continue
    attack_num +=1
    gl.set_value('attack_num', attack_num-1)
    adv, distortion, is_success, nqueries, theta_signopt = attack(xi, yi,
        targeted=args.targeted, query_limit=args.query, distortion=args.epsilon, args=args)
    adv = adv.clamp(0, 1)
    Linf_distortion = torch.norm((adv - xi).flatten(), np.inf)
    Linf_distortion = Linf_distortion.cpu().numpy()  
    
    L2_distances = torch.norm(adv - xi)  
    L2_distances = L2_distances.cpu().numpy()

    if theta_init is not None:
        match = (theta_signopt.astype(np.int32) == theta_init.astype(np.int32)).sum() / np.sum(abs(theta_signopt))
        print('sign matching rate between theta_init and theta_signopt:', match)

    if is_success :  #'''and nqueries !=0''' 
        stop_queries.append(nqueries)
        Linf_distances_list.append(Linf_distortion)
        L2_distances_list.append(L2_distances)
        success_num +=1

        ####################################################################################
        original_img_file = 'save/ours/original_images/original%s.jpg' % (i)   #保存原始图片    
        adv_img_file = 'save/ours/adv_images/adv%s.jpg' % (i)   #保存对抗样本 
        prub_img_file = 'save/ours/prub_images/adv%s.jpg' % (i)

        '''original_img_file = 'save/sign_opt/original_images/original%s.jpg' % (i)   #保存原始图片    
        adv_img_file = 'save/sign_opt/adv_images/adv%s.jpg' % (i)   #保存对抗样本 
        prub_img_file = 'save/sign_opt/prub_images/adv%s.jpg' % (i)'''       
        vutils.save_image(xi, original_img_file, normalize=True)
        vutils.save_image(adv, adv_img_file, normalize=True)      
        #vutils.save_image(torch.tensor(distortion*theta_signopt, dtype=torch.float).cuda(),prub_img_file, normalize=True)
        vutils.save_image(adv - xi,prub_img_file, normalize=True)       
        ####################################################################################

    if args.targeted == False:
        r_count = (torch.max(amodel.predict(adv),0)[1]==yi).nonzero().shape[0]
        clean_count = (torch.max(amodel.predict(xi),0)[1]==yi).nonzero().shape[0]
        total_r_count += r_count
        total_clean_count += clean_count
        total_distance += utils.distance(adv,xi,norm=norm.lower())

num_queries = amodel.get_num_queries()
'''succ = 0

for ele in range(0, len(successes)):
    succ = succ + successes[ele]

avg_success = succ / len(successes)'''

avg_success = success_num / attack_num
#logging.info(i, total_r_count, total_clean_count)
logging.info("="*10)
logging.info(f"clean count:{total_clean_count}")
logging.info(f"acc under attack count:{total_r_count}")
#logging.info(f"avg total queries used:{num_queries}")
#logging.info(f"avg stop queries used:{np.mean(stop_queries)}")
logging.info(f"avg distortion:{np.mean(Linf_distances_list)}")
######################
total_zero_num = gl.get_value('total_zero_num')
logging.info(f"average total_zero_num:{total_zero_num / success_num}")
under_thold = gl.get_value('under_thold')
logging.info(f"L2 under_thold rate:{under_thold / success_num}")
###################### 
logging.info(f"avg stop queries used:{np.mean(stop_queries)}")
logging.info(f"avg success:{avg_success}")

savefile = '%s.pth' % (
    args.model)
torch.save({'succs': under_thold /success_num, 'queries': stop_queries,'l2_norms': Linf_distances_list}, savefile)
print(Linf_distances_list)
print("median linf {:.4f}".format(np.median(Linf_distances_list)))
print("median l2 {:.4f}".format(np.median(L2_distances_list)))