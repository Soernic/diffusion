from pc_with_gradients import *
from time import time
import pandas as pd


def set_up_classifier_diffusion(args):
    """
    Straight forward. Sets up the classifier, the diffusion model, and the second classifier. 
    Returns in a dictionary.
    """

    classifier = ClassifierWithGradients(args)
    diffusion = DiffusionWithGradients(args, classifier)
    second_classifier = None

    if args.ckpt_second_classifier is not None:
        # Messy, but I think it works..
        args.ckpt_classifier = args.ckpt_second_classifier
        args.categories_classifier = args.categories_second_classifier
        second_classifier = ClassifierWithGradients(args)

    models = {'classifier': classifier, 'diffusion': diffusion, 'second_classifier': second_classifier}
    return models


def generate_batch(args, models: dict, y: torch.Tensor, q=False) -> List[GradientPointCloud]:
    """
    Generates a batch with parameters from args and returns it, guiding towards a class
    """

    if not q: print(f'Batch size: {args.batch_size} | Gradient scale: {args.gradient_scale} | Desired class: {models['classifier'].mask[args.desired_class]}')
    return models['diffusion'].sample(y=y)


def save_clouds(batch, file_loc: str = None):
    """
    Save the clouds somewhere
    """
    if file_loc is not None:
        os.makedirs(file_loc, exist_ok=True)
        # pass
        # TODO add here with local folders and all that
        # Idea is to append batch_cloud_{idx} to the file_loc with os.path.join or something
        # Need to set up with stream.py so we can view them as well thoguh, otherwise not useful
    else:
        ValueError('You have to select a subfolder for the pcs folder in the file_loc variable variable in the main script below.')

    for idx, cloud in enumerate(batch): 
        cloud.save(os.path.join(file_loc, f'batch_cloud_{idx}')) # TODO: Add local folder for each package of plots later
    return 


########################## CODE FOR PLOT 1 #################################
def save_cloud(cloud, loc, name):
    """
    Takes only one cloud
    """
    cloud.save(os.path.join(loc, name))

def generate_s_batch(args, file_loc, y):
    # set_trace()
    os.makedirs(os.path.join('pcs', file_loc), exist_ok=True)
    # s_range = [0, 1, 10, 100, 1000, 5000]
    # s_range = [0] + [5**i for i in range(3, 6)]
    s_range = [0, 500, 2000, 5000, 10000]

    for s in s_range:
        name = str(f's_{s:05d}')
        args.gradient_scale = s
        # seed_all(args.seed)
        models = set_up_classifier_diffusion(args)
        # set_trace()
        seed_all(args.seed) # Since different classes in classifier, it behaves weird. This hard resets it.
        cloud = generate_batch(args, models, y)[0]
        save_cloud(cloud, file_loc, name)


def plot_1(args):
        prefix = './relevant_checkpoints'
        file_locs = ['cl1', 'cl2', 'cl3']
        # TODO: Update the file to be the actually best one in the training for the mean
        classifier_names = ['cl_2_max_100.pt', 'cl_all_max_100.pt', 'cl_all_mean_100.pt'] 
        classifier_paths = [os.path.join(prefix, c) for c in classifier_names]
        classifier_categories = [['airplane', 'chair'], ['all'], ['all']]

        for file_loc, classifier_path, classifier_category in zip(file_locs, classifier_paths, classifier_categories):
            # seed_all(args.seed)
            args.ckpt_classifier = classifier_path
            args.categories_classifier = classifier_category
            # set_trace()
            generate_s_batch(args, file_loc, y)
########################## CODE FOR PLOT 1 #################################


# ########################## CODE FOR PLOT 5 #################################
# def classify_clouds(pcs: List[GradientPointCloud]):
#     labels = [pc.classify(models['classifier']) for pc in pcs]
#     return labels


# def compute_percentage(labels: List[str]):
#     return labels.count('airplane') / float(len(labels))


# def plot_5(args, filename):

#     # Hyperparams
#     EXP_RANGE = 16          # 16
#     BATCH_SIZE = 64         # 64
#     RANGE = 8               # 8


#     s_vals = [0] + [2**i for i in range(EXP_RANGE)]
#     # s_vals = np.arange(200, step=100)
#     classifier_acrynym = ['cl1', 'cl2', 'cl3']
#     vals = pd.DataFrame(columns=(['s_vals'] + classifier_acrynym)) # Append the percentages to the cls afterwards
#     vals['s_vals'] = s_vals

#     args.batch_size = BATCH_SIZE
#     prefix = './relevant_checkpoints'
#     classifier_names = ['cl_2_max_100.pt', 'cl_all_max_100.pt', 'cl_all_mean_100.pt'] 
#     classifier_paths = [os.path.join(prefix, c) for c in classifier_names]
#     classifier_categories = [['airplane', 'chair'], ['airplane', 'chair'], ['all']]

#     # For each classifier
#     for classifier_path, classifier_category, acronym in zip(classifier_paths, classifier_categories, classifier_acrynym):
#         args.ckpt_classifier = classifier_path
#         args.categories_classifier = classifier_category
#         percentages = list()

#         pcs = list()

#         # .. loop through the s-range (x-axis)
#         for s in tqdm(s_vals, acronym):
#             args.gradient_scale = s
#             models = set_up_classifier_diffusion(args)

#             # .. generate a bunch of clouds
#             for iter in range(RANGE):
#                 batch = generate_batch(args, models, y, q=True)
#                 pcs.extend(batch)
#                 # set_trace()

#             # .. and classify them
#             labels = classify_clouds(pcs)
#             percentage = compute_percentage(labels)

#             # .. to compute the percentage of airplanes (y-axis)
#             percentages.append(percentage)
#         vals[acronym] = percentages
#         # set_trace()

#     os.makedirs('plots/plot5', exist_ok=True)

#     # TODO: Simplify file name
#     vals.to_csv(f'plots/plot5/{filename}_{EXP_RANGE}_{BATCH_SIZE}.csv', index=False)
# ########################## CODE FOR PLOT 5 #################################




if __name__ == '__main__':
    # Arguments
    
    # Things you should not touch. plz
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
    parser.add_argument('--ret_traj', type=eval, default=False, choices=[True, False])
    parser.add_argument('--num_batches', type=int, default=1) # pcs generated = num_batches x batch_size
    parser.add_argument('--categories', type=str_list, default=['airplane', 'chair'])

    # Things you can touch. TODO: Note, if you change the ckpt_classifier, make sure the categories_classifier are correct.
    parser.add_argument('--ckpt', type=str, default='./relevant_checkpoints/ckpt_base_1M.pt')
    parser.add_argument('--ckpt_classifier', type=str, default='./relevant_checkpoints/cl_2_max_100.pt')
    parser.add_argument('--categories_classifier', type=str_list, default=['airplane', 'chair'])
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--gradient_scale', type=float, default=1) # noted "s" usually
    parser.add_argument('--desired_class', type=int, default=0)

    # The classifier could be guiding towards what it "thinks" is an airplane.
    # Input a path to a second classifier here if you want. 
    parser.add_argument('--ckpt_second_classifier', type=str, default=None)
    parser.add_argument('--second_categories_classifier', type=str_list, default=['airplane', 'chair'])

    # Don't touch this either
    args = parser.parse_args()
    seed_all(args.seed) 
    y = torch.ones(args.batch_size, dtype=torch.long).to(args.device) * args.desired_class
    models = set_up_classifier_diffusion(args) # Get the classifier


    ########################## CODE FOR PLOT 1 #################################
    # For plot 1. Open in streamlit with custom port from terminal like so:
    # > streamlit run stream.py --server.port xxxx
    # If it keeps loading, select a different --server.port. Default is 8501, but it might be occupied
    # Default savepaths are cl1, cl2, cl3 in pcs, so open those. 
    # Adjust view height etc. with the sliders
    # You can press "Save all plots as PNG" to save them to the library
    plot_1(args)
    ########################## CODE FOR PLOT 1 #################################


    ########################## CODE FOR PLOT 5 #################################
    # Plot 5 computes a csv-file with the airplane prediction ratio for each classifier as we vary s
    # It computes each point over 512 samples, so N should be plenty high.
    # Takes about 45 minutes to run on a v100
    # Please be aware that this only runs for one diffusion model
    # If you rerun and change the diffusion model, please change the name of the file
    # That way you don't overwrite the existing .csv
    # plot_5(args, '100_steps')
    ########################## CODE FOR PLOT 5 #################################


    





    



