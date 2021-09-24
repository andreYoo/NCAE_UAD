import torch
import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from importlib import reload
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps
import cv2
reload(plt)
colour_code = ['b', 'g', 'r', 'c', 'm', 'y', 'k','tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']



def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def plot_images_grid(filetitle,x: torch.tensor, title, nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(filetitle, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()

def plot_multiple_images_grid(filetitle,x: [torch.tensor], title,subtitle=None, nrow=8, padding=2, normalize=False, pad_value=0,num_images=1):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""
    _len = len(x) #num of images
    if subtitle==None:
        sub_title= ['Input','Recon','Gen']
    else:
        sub_title=subtitle
    f = plt.figure()

    for i in range(_len):
        npgrid = make_grid(x[i], nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).cpu().numpy()
        f.add_subplot(1, _len, i + 1)
        plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(sub_title[i])

    if not (title == ''):
        plt.suptitle(title)

    plt.savefig(filetitle, bbox_inches='tight', pad_inches=0.1,dpi=600)
    plt.clf()
    plt.close()


def error_bar(filename,images,recon):
    _shape = np.shape(images)
    _sorted_idx = np.argsort(recon)
    _len = _shape[0]
    labels = np.arange(_len)*100
    width = 50
    images = images.reshape(_len, _shape[2], _shape[3])
    plt.figure(figsize=(15, 5))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    def offset_image(img, x, y, ax):
        im = OffsetImage(img, zoom=1.0)
        im.image.axes = ax
        x_offset = 20
        ab = AnnotationBbox(im, (y, x), xybox=(0, x_offset), frameon=False,
                            xycoords='data', boxcoords="offset points", pad=0)
        ax.add_artist(ab)
    plt.ylim([0.0,0.47])
    plt.bar(x=labels[0:10]-50, width=width, height=recon[_sorted_idx[0:10]], color='r', align='center', alpha=0.8)
    plt.bar(x=labels[10:20]-50, width=width, height=recon[_sorted_idx[len(_sorted_idx)-10:len(_sorted_idx)]], color='b', align='center', alpha=0.8)
    for i in range(20):
        if i < 10:
            offset_image(images[_sorted_idx[i]],recon[_sorted_idx[i]], labels[i]-50, ax=plt.gca())
        else:
            offset_image(images[_sorted_idx[len(_sorted_idx)-20+i]],recon[_sorted_idx[len(_sorted_idx)-20+i]], labels[i]-50, ax=plt.gca())

    plt.axhline(y=np.mean(recon), color='black', linestyle=':')
    plt.savefig(filename,dpi=600)
    plt.clf()
    plt.close()


# def error_bar(filename,images,recon):
#     _shape = np.shape(images)
#     _sorted_idx = np.argsort(recon)
#     _len = _shape[0]
#     labels = np.arange(_len)*120
#     width = 60
#     images = images.reshape(_len, _shape[2], _shape[3])
#     plt.figure(figsize=(20, 4))
#     plt.tick_params(
#         axis='x',  # changes apply to the x-axis
#         which='both',  # both major and minor ticks are affected
#         bottom=False,  # ticks along the bottom edge are off
#         top=False,  # ticks along the top edge are off
#         labelbottom=False)  # labels along the bottom edge are off
#
#     def offset_image(img, x, y, ax):
#         im = OffsetImage(img, zoom=0.5)
#         im.image.axes = ax
#         x_offset = 10
#         ab = AnnotationBbox(im, (y, x), xybox=(0, x_offset), frameon=False,
#                             xycoords='data', boxcoords="offset points", pad=0)
#         ax.add_artist(ab)
#
#     plt.ylim([0.7,1.0])
#     plt.bar(x=labels[_sorted_idx[0:5]]-60, width=width, height=recon[_sorted_idx[0:5]], color='r', align='center', alpha=0.8)
#     plt.bar(x=labels[_sorted_idx[5:]]-60, width=width, height=recon[_sorted_idx[5:]], color='b', align='center', alpha=0.8)
#     for i in range(_len):
#         offset_image(images[i],recon[i], labels[i]-60, ax=plt.gca())
#
#     plt.axhline(y=np.mean(recon), color='black', linestyle=':')
#     plt.savefig(filename,dpi=600)
#     plt.clf()
#     plt.close()


def TSNE_distributions_plotting(filename,features,leg=['Latent features','Sampling noise']):
    tsne = TSNE(n_jobs=4)
    len_feature = len(features)
    _num_feature = np.shape(features[0])[0]
    for i in range(len_feature):
        if i==0:
            total_features=tsne.fit_transform(features[i])
            total_label = np.ones([_num_feature])*i
        else:
            total_features = np.concatenate((total_features,tsne.fit_transform(features[i])),axis=0)
            total_label = np.concatenate((total_label,np.ones([_num_feature])*i),axis=0)

    #total_features,total_label= unison_shuffled_copies(total_features,total_label)

    results_tsne = total_features
    # results_tsne = total_lat
    num_dist = len_feature
    plt.figure()
    plt.grid(True)
    for j in range(num_dist):
        plt.scatter(results_tsne[total_label==j, 0], results_tsne[total_label==j, 1], c=colour_code[j], s=40, marker='.',label=leg[j])
    plt.legend(loc='lower left',fontsize=14)
    plt.savefig(filename)
    plt.clf()
    plt.clf()
    plt.close()
    return results_tsne,total_label



def scatter_feature_with_entropy(filename,features,labels):
    tsne = TSNE(n_jobs=4)
    len_feature = len(features)
    results_tsne = tsne.fit_transform(features)
    # results_tsne = total_lat
    num_label = int(max(labels))+1
    print('Done!')
    plt.figure()
    if num_label==1:
        plt.scatter(results_tsne[:, 0], results_tsne[:, 1], c='blue', s=5, marker='.')
    elif num_label==2:
        plt.scatter(results_tsne[labels==0, 0], results_tsne[labels==0, 1], c='blue', s=5, marker='.')
        plt.scatter(results_tsne[labels==1, 0], results_tsne[labels==1, 1], c='red', s=5, marker='.')
    else:
        for i in range(num_label):
            plt.scatter(results_tsne[labels==i, 0], results_tsne[labels==i, 1], c=colour_code[i], s=5, marker='.')
    plt.scatter(results_tsne[labels==-1, 0],results_tsne[labels==-1, 1],c='black',s=50,marker='+')
    plt.savefig(filename)
    plt.close()
    plt.clf()


def TSNE_feature_plotting(filename,features,labels,centre):
    tsne = TSNE(n_jobs=4)
    len_feature = len(features)
    clabels = -1*np.ones(len(centre))
    features = np.concatenate((features,centre),axis=0)
    labels = np.concatenate((labels,clabels),axis=0)
    results_tsne = tsne.fit_transform(features)
    # results_tsne = total_lat
    num_label = int(max(labels))+1
    print('Done!')
    plt.figure()
    if num_label<=2:
        plt.scatter(results_tsne[labels==0, 0], results_tsne[labels==0, 1], c='blue', s=5, marker='.')
        plt.scatter(results_tsne[labels==1, 0], results_tsne[labels==1, 1], c='red', s=5, marker='.')
    else:
        for i in range(num_label):
            plt.scatter(results_tsne[labels==i, 0], results_tsne[labels==i, 1], c=colour_code[i], s=5, marker='.')
    plt.scatter(results_tsne[labels==-1, 0],results_tsne[labels==-1, 1],c='black',s=50,marker='+')
    plt.savefig(filename)
    plt.close()
    plt.clf()
    return results_tsne



def feature_plotting(filename,features,labels,centre):
    results_tsne = features
    clabels = -1*np.ones(len(centre))
    labels = np.concatenate((labels,clabels),axis=0)
    # results_tsne = total_lat
    num_label = int(max(labels))+1
    print('Done!')
    plt.figure()
    if num_label<=2:
        plt.scatter(results_tsne[labels==0, 0], results_tsne[labels==0, 1], c='blue', s=5, marker='.')
        plt.scatter(results_tsne[labels==1, 0], results_tsne[labels==1, 1], c='red', s=5, marker='.')
    else:
        for i in range(num_label):
            plt.scatter(results_tsne[labels==i, 0], results_tsne[labels==i, 1], c=colour_code[i], s=5, marker='.')
    plt.scatter(results_tsne[labels==-1, 0],results_tsne[labels==-1, 1],c='black',s=50,marker='+')
    plt.savefig(filename)
    plt.clf()
    return results_tsne

def entropy_based_plotting(filename,features,entropy,centre):
    results_tsne = features
    entr = entropy(features).sum(axis=1)
    _len = len(entropy)
    # results_tsne = total_lat
    print('Done!')
    plt.figure()
    plt.scatter(results_tsne[:,0],results_tsne[:,1], c=entr, s=10, marker='.')
    plt.savefig(filename)
    plt.clf()
    return results_tsne


def scatter_plot(data_loader,models,device,c=None,filename=None,AE=False):
    tsne = TSNE(n_jobs=4)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels, _, index = data
            inputs = inputs.to(device)
            # Update network parameters via backpropagation: forward + backward + optimize
            if AE==True:
                _,lat = models(inputs,L_vis=True)
            else:
                lat = models(inputs)
            if i == 0:
                total_lat = lat.detach().cpu().numpy()
                total_index = index.detach().cpu().numpy()
                total_labels = labels.detach().cpu().numpy()
            else:
                total_lat = np.concatenate((total_lat, lat.detach().cpu().numpy()))
                total_index = np.concatenate((total_index, index.detach().cpu().numpy()))
                total_labels = np.concatenate((total_labels, labels.detach().cpu().numpy()))
        if c is not None:
            total_lat  = np.concatenate((total_lat,c.view(1,-1).detach().cpu().numpy()))


        results_tsne = tsne.fit_transform(total_lat)
        # results_tsne = total_lat
        print('Done!')
        plt.figure()
        plt.scatter(results_tsne[:, 0], results_tsne[:, 1], c='green', s=5, marker='.')
        if c is not None:
            plt.scatter(results_tsne[-1, 0],results_tsne[-1, 1],c='black',s=30,marker='+')
        plt.savefig(filename)
    plt.clf()



def ab_no_scatter_plot(data_loader,models,device,c=None,filename=None,AE=False):
    tsne = TSNE(n_jobs=4)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels, _, index = data
            inputs = inputs.to(device)
            # Update network parameters via backpropagation: forward + backward + optimize
            if AE == True:
                _, lat = models(inputs, L_vis=True)
            else:
                lat = models(inputs)
            if i == 0:
                total_lat = lat.detach().cpu().numpy()
                total_index = index.detach().cpu().numpy()
                total_labels = labels.detach().cpu().numpy()
            else:
                total_lat = np.concatenate((total_lat, lat.detach().cpu().numpy()))
                total_index = np.concatenate((total_index, index.detach().cpu().numpy()))
                total_labels = np.concatenate((total_labels, labels.detach().cpu().numpy()))

        if c is not None:
            total_lat  = np.concatenate((total_lat,c.view(1,-1).detach().cpu().numpy()))
            total_index = np.concatenate((total_index,[-1]))
            total_labels = np.concatenate((total_labels, [-1]))

        results_tsne = tsne.fit_transform(total_lat)
        # results_tsne = total_lat
        print('Done!')
        plt.figure()
        plt.scatter(results_tsne[total_labels==0, 0], results_tsne[total_labels==0, 1], c='blue', s=5, marker='.')
        plt.scatter(results_tsne[total_labels==1, 0], results_tsne[total_labels==1, 1], c='red', s=5, marker='.')
        if c is not None:
            plt.scatter(results_tsne[-1, 0],results_tsne[-1, 1],c='black',s=30,marker='+')
        plt.savefig(filename)
    plt.clf()

def ab_no_scatter_plot_with_gan(data_loader,models,generator,device,c=None,filename=None,AE=False):
    tsne = TSNE(n_jobs=4)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels, _, index = data
            inputs = inputs.to(device)
            # Update network parameters via backpropagation: forward + backward + optimize
            if AE == True:
                _, lat = models(inputs, L_vis=True)
            else:
                lat = models(inputs)
            noise = torch.FloatTensor(lat.size()).normal_(0, 1).cuda()
            _,fake = models(generator(noise),L_vis=True)

            if i == 0:
                total_lat = lat.detach().cpu().numpy()
                total_index = index.detach().cpu().numpy()
                total_labels = labels.detach().cpu().numpy()
                total_fake = fake.detach().cpu().numpy()

            else:
                total_lat = np.concatenate((total_lat, lat.detach().cpu().numpy()))
                total_index = np.concatenate((total_index, index.detach().cpu().numpy()))
                total_labels = np.concatenate((total_labels, labels.detach().cpu().numpy()))
                total_fake = np.concatenate((total_fake, fake.detach().cpu().numpy()))

        total_lat = np.concatenate((total_lat, total_fake))
        fake_label = np.zeros([len(total_fake)])+-2
        total_labels = np.concatenate((total_labels, fake_label))


        if c is not None:
            total_lat  = np.concatenate((total_lat,c.view(1,-1).detach().cpu().numpy()))
            total_index = np.concatenate((total_index,[-1]))
            total_labels = np.concatenate((total_labels, [-1]))


        results_tsne = tsne.fit_transform(total_lat)
        # results_tsne = total_lat
        print('Done!')
        plt.figure()
        plt.scatter(results_tsne[total_labels==0, 0], results_tsne[total_labels==0, 1], c='blue', s=5, marker='.')
        plt.scatter(results_tsne[total_labels==1, 0], results_tsne[total_labels==1, 1], c='red', s=5, marker='.')
        plt.scatter(results_tsne[total_labels==-2,0],results_tsne[total_labels==-2,1],c='black',s=5,marker='o')
        if c is not None:
            plt.scatter(results_tsne[-1, 0],results_tsne[-1, 1],c='yellow',s=60,marker='+')
        plt.savefig(filename)
    plt.clf()


def image_scatter(filename,images,feature_2d,labels):
    _shape = np.shape(images)
    _len = _shape[0]
    fig, ax = plt.subplots()
    ax.grid(True)
    images = images.reshape(_len,_shape[2],_shape[3])
    ax.scatter(feature_2d[labels==0,0],feature_2d[labels==0,1],c=colour_code[0])
    ax.scatter(feature_2d[labels==1,0],feature_2d[labels==1,1],c=colour_code[1])
    for _s in range(_len):
        img= OffsetImage(images[_s],zoom=0.5)
        ab = AnnotationBbox(img,(feature_2d[_s,0], feature_2d[_s,1]), frameon=False)
        ax.add_artist(ab)
    plt.savefig(filename)
    plt.close()
    plt.clf()

def image_scatter_with_coloured_boundary(filename,images,feature_2d,labels):
    _shape = np.shape(images)
    _len = _shape[0]
    fig, ax = plt.subplots()
    images = images.reshape(_len,_shape[2],_shape[3])
    ax.scatter(feature_2d[labels==0,0],feature_2d[labels==0,1],c=colour_code[0])
    ax.scatter(feature_2d[labels==1,0],feature_2d[labels==1,1],c=colour_code[1])
    ax.grid(True)
    for _s in range(_len):
        if _s < 128:
            formatted = (images[_s] * 255 / np.max(images[_s])).astype('uint8')
            formatted = np.transpose(np.array([formatted,formatted,formatted]),(1,2,0))
            _img = Image.fromarray(formatted)
            _img = ImageOps.expand(_img, border=2, fill='blue')
            img= OffsetImage(_img,zoom=0.5)
            ab = AnnotationBbox(img,(feature_2d[_s,0], feature_2d[_s,1]), frameon=False)
            ax.add_artist(ab)
        else:
            formatted = (images[_s] * 255 / np.max(images[_s])).astype('uint8')
            formatted = np.transpose(np.array([formatted,formatted,formatted]),(1,2,0))
            _img = Image.fromarray(formatted)
            _img = ImageOps.expand(_img, border=2, fill='green')
            img= OffsetImage(_img,zoom=0.5)
            ab = AnnotationBbox(img,(feature_2d[_s,0], feature_2d[_s,1]), frameon=False)
            ax.add_artist(ab)

    plt.savefig(filename,dpi=600)
    plt.close()
    plt.clf()
