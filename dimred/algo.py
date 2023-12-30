import numpy as np
import pickle
from sklearn.decomposition import PCA, FastICA, SparsePCA, DictionaryLearning, SparseCoder, NMF
from dimred.data import Transform, DataLoader
import pandas as pd
from pychubby.detect import LandmarkFace
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile, AbsoluteMove, Action, Lambda
import colorsys
import matplotlib.pyplot as plt
import os
import glob
import cv2
import itertools
import multiprocessing as mp


face_kp_n = np.loadtxt("../alphapose/face_kp_n_scaled.txt", delimiter=" , ")     
inp_path_n = "../alphapose/alpha_in/000_n_scaled.jpg"


class Helper:
    def __init__(self, face_part=None):
        self.face_part=face_part
        pass

    def part_to_68_kp(self, loadings, parts, part):

        mid = loadings.shape[1]//2
        loadings_temp = np.zeros((loadings.shape[0], 136))
        a = parts[part][0]
        b = parts[part][1]
        # print(loadings.shape, a, b, loadings_temp.shape, mid)
        loadings_temp[:, a:b], loadings_temp[:, (a+68):(b+68)] = loadings[:, :mid], loadings[:, mid:]
        return loadings_temp


    def dataframe_neg_to_pos(self, df):
        df = pd.concat([df[df>=0], -df[df<0]], axis=1)
        df.fillna(0, inplace=True)
        return df


    def find_minvar_DL_path(self, model_prefix, var_threshold):
        temp_path = ""
        temp_var = 0
        flag = 0
        for path in glob.glob(model_prefix+"*tmi/", recursive=True):
            try:
                var = pd.read_csv(path+"train_var.csv", index_col=0).iloc[0,0]
                if var>=var_threshold:
                    if flag==0:
                        temp_var = var
                        temp_path = path
                        flag = 1
                    elif var<temp_var:
                        temp_var = var
                        temp_path = path
            except:
                pass
        return temp_path


    def find_minvar_nmf_path(self, model_prefix, var_threshold):
        temp_path = ""
        temp_var = 0
        flag = 0
        for path in glob.glob(model_prefix+"*alpha_W/", recursive=True):
            var = pd.read_csv(path+"full_train_var.csv", index_col=0).iloc[0,0]
            if var>=var_threshold:
                if flag==0:
                    temp_var = var
                    temp_path = path
                    flag = 1
                elif var<temp_var:
                    temp_var = var
                    temp_path = path
        return temp_path



    def plot_corr(self, X_transform, corr_save_path, parts):
        f = plt.figure(figsize=(19, 15))
        plt.matshow(X_transform.corr(), fignum=f.number)

        # print(X_transform.columns)

        colors_list = []
        for part in parts:
            # if part=="left_eyebrow_plus_right_eyebrow":
            #     substr = "eyebrow"
            # elif part=="left_eye_plus_right_eye":
            #     substr = "eyes"
            # else:
            substr = str(part)+"_" # Placing an underscore because e.g., left_eyebrow and left_eye both have common substring
            color_cols = [s for s in X_transform.columns if substr in s]
            # print(len(color_cols))
            for p in range(len(color_cols)):
                colors_list.append(parts[part][2])

        print(colors_list)

        # print("Length of colors : ", len(colors_list), X_transform.shape)

        # plt.xticks(range(5), X_transform.select_dtypes(['number']).columns[:5], fontsize=14, rotation=45, color='red')
        # plt.yticks(range(5), X_transform.select_dtypes(['number']).columns[:5], fontsize=14, color='blue')
        plt.xticks(range(X_transform.select_dtypes(['number']).shape[1]), X_transform.select_dtypes(['number']).columns, fontsize=14, rotation=45)
        plt.yticks(range(X_transform.select_dtypes(['number']).shape[1]), X_transform.select_dtypes(['number']).columns, fontsize=14)
        for ticklabel1, ticklabel2, tickcolor in zip(plt.gca().get_xticklabels(), plt.gca().get_yticklabels(), colors_list):
            ticklabel1.set_color(tickcolor)
            ticklabel2.set_color(tickcolor)


        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16);
        plt.savefig(corr_save_path, dpi=300)
        plt.close()




    def plot_hist(self, X_transform, pca_X_transform, path_to_fig, label):
        # df = pd.read_csv(path_to_X_transform, index_col=0)
        # X_transform = df.to_numpy()
        plt.rcParams["figure.figsize"] = (20,10)
        non_zeros = np.count_nonzero(X_transform, axis=1)

        d = non_zeros.copy()
        # d = d[d>0]
        # print(d)
        # generate histogram

        fig, ax = plt.subplots()

        # freq, bins, patches = plt.hist(d, edgecolor='white', label='d', bins=np.unique(d))
        # print(freq)
        # a histogram returns 3 objects : n (i.e. frequncies), bins, patches


        lt = list(np.unique(d))
        lt.append(np.amax(d)+1)
        bins_value = lt.copy()
        bins_value = np.array(lt)

        freq, bins, patches = plt.hist(d, edgecolor='white', label=label, bins=bins_value, weights=100*np.ones(len(d))/len(d), cumulative=True)


        # freq = 100*freq/np.sum(freq)
        # print("bins: ", bins)
        # print("freq: ", freq)

        lab = []
        for val in bins:
            lab.append(str(val))

        # x coordinate for labels
        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        offset = bin_centers[-1]
        # print(bin_centers)
        ax.set_xticks(bins_value+0.5)
        ax.set_xticklabels(lab)

        past_bins_value = bins_value
        past_lab = lab


        # print(bin_centers)

        n = 0
        for fr, x, patch in zip(freq, bin_centers, patches):
            # print(x)
            height = float(freq[n])
            plt.annotate("{:.2f}%".format(height),
                        xy = (x, int(height)),             # top left corner of the histogram bar
                        xytext = (0,0.2),             # offsetting label position above its bar
                        textcoords = "offset points", # Offset (in points) from the *xy* value
                        ha = 'center', va = 'bottom', size=8
                        )
            # default font size in 10
            n = n+1

        # plt.show()
        # df = pd.read_csv(path_to_pca_X, index_col=0)
        # X_transform = df.to_numpy()
        non_zeros = np.count_nonzero(pca_X_transform, axis=1)

        d = non_zeros.copy()
        temp_d = d.copy()
        # fig, ax = plt.subplots()
        d = d+int(1.5*np.amax(d))
        lt = list(np.unique(d))
        lt.append(np.amax(d)+1)
        # temp = np.zeros(int(1.5*past_bins_value.shape))
        bins_value = lt.copy()
        bins_value = np.array(lt)
        # bins_value = np.concatenate((temp, bins_value))
        
        freq, bins, patches = plt.hist(d, color='violet', edgecolor='white', label='PCA', bins=bins_value, weights=100*np.ones(len(d))/len(d), cumulative=True)

        # freq = 100*freq/np.sum(freq)
        # print("bins: ", bins)
        # print("freq: ", freq)

        lab = []
        for val in bins:
            lab.append(str(val))

        # x coordinate for labels
        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        print(bin_centers)
        # print(bin_centers)
        print("Past lab and lab is :", past_lab, lab, str(temp_d))
        ax.set_xticks(np.concatenate((past_bins_value, bins_value))+0.5)
        print(past_bins_value, bins_value, past_lab, temp_d)

        # Below for part face model
        # ax.set_xticklabels(np.concatenate((past_lab, [str(temp_d[0]), str(temp_d[0])])))
        # Below for full face model
        ax.set_xticklabels(np.concatenate((past_lab, [str(temp_d[0]), str(temp_d[0]), str(temp_d[0])])))

        n = 0
        for fr, x, patch in zip(freq, bin_centers, patches):
            # print(x)
            height = float(freq[n])
            plt.annotate("{:.2f}%".format(height),
                        xy = (x, int(height)),             # top left corner of the histogram bar
                        xytext = (0,0.2),             # offsetting label position above its bar
                        textcoords = "offset points", # Offset (in points) from the *xy* value
                        ha = 'center', va = 'bottom', size=8
                        )
            n = n+1

        plt.legend(loc='upper left')
        plt.title(self.face_part)
        # plt.xlabel("No. of Components used per sample's Frequency based Histogram\nOur Algorithm used atmost 3 components for 100% of the samples\nwhile PCA used all 20 components for every sample")
        plt.xlabel("No. of Components Used")        
        # fig.subplots_adjust(bottom=0.200)
        fig.savefig(path_to_fig, dpi=300)
        # plt.show()
        plt.close()

    def draw_keypoints(self, kps, save_path=None, rad=3):
        maxx = np.amax(kps[:, 0])
        maxy = np.amax(kps[:, 1])
        img = np.zeros((int(1.25*maxy), int(1.25*maxx)))
        for x, y in zip(kps[:, 0], kps[:, 1]):
            img = cv2.circle(img, (int(x), int(y)), rad, (255, 255, 255), -1) 

        # cv2.imshow("KP_Image", img)
        # cv2.waitKey(0)
        if save_path is not None:
            cv2.imwrite(save_path, img)

    def var_explained(self, X, X_approx):
        # X and X_approx need to be numpy array
        return 100 - 100*(np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2)

    def img_folder_to_video_writer(self, path_to_img_dir, out_dir, fps):
        img_array = []
        for filename in glob.glob(path_to_img_dir+"*.jpg"):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        out = cv2.VideoWriter(out_dir,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def get_N_Colors(self, N):
        r = np.arange(0, 255, 256/N, dtype=np.int16)
        np.random.shuffle(r)
        g = np.arange(0, 255, 256/N, dtype=np.int16)
        np.random.shuffle(g)
        b = np.arange(0, 255, 256/N, dtype=np.int16)
        np.random.shuffle(b)
        color = []
        for i in range(N):
            # print(hex(r[i]), hex(g[i]), hex(b[i]))
            u = str(hex(r[i]))[2:]
            if len(u)==1:
                u = "0"+u
            v = str(hex(g[i]))[2:]
            if len(v)==1:
                v = "0"+v
            w = str(hex(b[i]))[2:]
            if len(w)==1:
                w = "0"+w
            color.append("#"+u+v+w)
        return color

    def get_N_HexCol(self, N=5):
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append('#%02x%02x%02x' % tuple(rgb))
        return hex_out


    def plot(self, x_list, y_list, legend_list, x_label, y_label, title, fig_path):
        # fig_path provide in png
        # color_list = self.get_N_HexCol(len(x_list))
        print(self.get_N_HexCol(len(x_list)))
        color_list = self.get_N_Colors(len(x_list))
        for i in range(len(x_list)):
            plt.plot(x_list[i], y_list[i], color=color_list[i], alpha=1, label=legend_list[i])
            try:
                x = []
                y = []
                for j in range(x_list[i].shape[0]):
                    if y_list[i][j]>95:
                        print(i, "95 reached")
                        x.append(x_list[i][j])
                        y.append(y_list[i][j])
                        x.append(x_list[i][j])
                        y.append(0)

                        break
                plt.plot(x, y, color=color_list[i], linestyle='dashed', marker='o', markerfacecolor=color_list[i], markersize=2)

                plt.annotate(str(x[0]), (x[1], y[1]))
            except Exception as e:
                print(str(e))
                pass
        plt.plot([-10,7000], [95,95], color="#000000", linestyle='dashed', marker='o', markerfacecolor="#000000", markersize=2)
        plt.annotate("95", (5, 95))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.xlim(0, 136)
        plt.title(title)

        plt.legend()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        # plt.show()
        

    def sort(self, X, axis, index, asc=False):
        #asc=True will sort in ascending order based on index
        #axis=1 will sort the array based on the row along the index
        if axis==1:
            X = X.T
        if asc==False:
            X[:, index] = -X[:, index]
        X = X[X[:, index].argsort()]
        if asc==False:
            X[:, index] = -X[:, index]
        if axis==1:
            X = X.T
        return X

    def L2_norm(self, X, Y):
        # X and Y are numpy arrays
        # returns the L2 norms between each row of X and each row of Y
        a = (np.diagonal(X@X.T).reshape(X.shape[0], 1))@np.ones((1, Y.shape[0]))
        b = np.ones((X.shape[0], 1))@(np.diagonal(Y@Y.T).reshape(1, Y.shape[0]))
        c = 2*X@Y.T
        return np.sqrt(a+b-c)

    def dot_product(self, X, Y):
        # X and Y are numpy arrays
        return X@Y.T


    def var_and_cumvar(self, X, loadings, X_transform=None, outdir=None, algo=None):
        # X is a numpy array of data
        # This functions calculates var and cumvar using pinv
        # rows of loadings represent normalized components

        var_arr = np.zeros((loadings.shape[0],2))
        var_arr[:,0] = np.arange(0, loadings.shape[0])

        h = Helper()

        for i in range(loadings.shape[0]):
            comp = loadings[i]
            comp = comp.reshape(1, loadings.shape[1])
            if X_transform is None:
                X_approx = (X@np.linalg.pinv(comp))@comp
            else:
                X_approx = (X_transform[:, i].reshape(X_transform.shape[0], 1))@(loadings[i, :].reshape(1, loadings.shape[1]))
            if algo=="pca":
                X_approx += np.mean(X, axis=0)
            var = 100*(1 - (np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2))
            var_arr[i, 1] = var
        # pd.DataFrame(var_arr).to_csv("temp_var_arr.csv")
        sorted_var_arr = h.sort(var_arr.copy(), 0, 1)

        cum_var_arr = np.zeros((loadings.shape[0],2))
        cum_var_arr[:,0] = sorted_var_arr[:,0]
        
        cum_comp_nonzero = np.zeros((loadings.shape[0], 1))
        for i in range(cum_var_arr.shape[0]):
            index = int(cum_var_arr[i,0])
            if X_transform is None:

                if i==0:
                    comp = loadings[index]
                    comp = comp.reshape(1, loadings.shape[1])
                else:
                    comp = np.concatenate((comp, loadings[index].reshape(1, loadings.shape[1])), axis=0)
                X_approx = (X@np.linalg.pinv(comp))@comp

            else:
                if i==0:
                    X_transformed_sorted = X_transform[:, index].reshape(X_transform.shape[0], 1)
                    comp = loadings[index]
                    comp = comp.reshape(1, loadings.shape[1])
                else:
                    X_transformed_sorted = np.concatenate((X_transformed_sorted, X_transform[:, index].reshape(X_transform.shape[0], 1)), axis=1)
                    comp = np.concatenate((comp, loadings[index].reshape(1, loadings.shape[1])), axis=0)
                X_approx = X_transformed_sorted@comp

            cum_comp_nonzero[i, 0] = np.count_nonzero(comp)
            if algo=="pca":
                X_approx+=np.mean(X, axis=0)
            var = 100*(1 - (np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2))
            cum_var_arr[i, 1] = var
    
        if outdir is not None:
            np.set_printoptions(suppress=True)
            np.savetxt(outdir+"var_arr.txt", var_arr, delimiter=' , ', fmt="%f")
            np.savetxt(outdir+"sorted_var_arr.txt", sorted_var_arr, delimiter=' , ', fmt="%f")
            np.savetxt(outdir+"cum_var_arr.txt", cum_var_arr, delimiter=' , ', fmt="%f")

        if outdir is not None:
            df = pd.DataFrame(np.concatenate((sorted_var_arr, cum_var_arr[:, 1].reshape(cum_var_arr.shape[0], -1)), axis=1), columns=['Components', 'Variance', 'Cum Variance'])
            df.to_csv(outdir + "var_list.csv")
            df = pd.DataFrame(np.concatenate((cum_comp_nonzero, cum_var_arr[:, 1].reshape(cum_var_arr.shape[0], -1)), axis=1), columns=['Non Zeros in V', 'Cum Variance'])
            df.to_csv(outdir + "var_list_nonzero.csv")           
        return var_arr, sorted_var_arr, cum_var_arr





    def vis(self, X, loadings_var, loadings_vis, outdir, face_part, X_transform=None, var_dir=None, scale=100):
        # if rtheta and cols_list are not None, meaning loadings is in rtheta format
        # inp_path_n = "/mnt/scratch2/csy207576/project/AU_Unsupervised/alphapose/alpha_in/000_n.jpg"
        a = face_part[0]
        b = face_part[1]
        trans = Transform()
        h = Helper()
        var_arr, _, _ = h.var_and_cumvar(X, loadings_var, X_transform, var_dir)

        # if rtheta:
        #     # if rtheta is not None then cols_list won't be empty
        #     d = DataLoader()
        #     loadings, _ = d.rtheta_to_delta(pd.DataFrame(loadings, columns = cols_list), [a,b])
        #     pd.DataFrame(loadings).to_csv(var_dir+"loadings_delta.csv")
        #     # print(loadings.columns)
        #     loadings = loadings.to_numpy()
        # Normalising loadings to view it

        for i in range(loadings_vis.shape[0]):
            if np.sum(loadings_vis[i])!= 0:
                loadings_vis[i] = loadings_vis[i]/np.linalg.norm(loadings_vis[i])

        h = Helper()

        for i in range(0, loadings_vis.shape[0]):

            # if i!=4:
            #     continue
            # prod = (X_transform[:, i].reshape(X_transform.shape[0], 1)@(loadings_var[i, :].reshape(1, (b-a)*4)))
            # temp = prod[:, :2*(b-a)] - prod[:, 2*(b-a):]
            
            # std_dev = np.zeros(136)
            # temp_std = np.std(temp, axis=0)
            # std_dev[a:b], std_dev[(a+68):(b+68)] = temp_std[:(b-a)], temp_std[(b-a):] 
            # temp = np.mean(temp, axis=0)
            # loadings_vis[i, :] = temp.copy()

            comp = np.zeros(136)
            comp[a:b] = loadings_vis[i, :(b-a)]
            comp[(a+68):(b+68)] = loadings_vis[i, (b-a):]
            comp = comp.reshape(1, 136)
            # outdir1 = outdir+"{:.2f}".format(var_arr[i,1])+"_"+str(i)+"/"
            # try:
            #     os.makedirs(outdir1)
            # except:
            #     pass
            # for j in range(temp.shape[0]):
            #     comp = np.zeros(136)
            #     comp[a:b] = temp[j, :(b-a)]
            #     comp[(a+68):(b+68)] = temp[j, (b-a):]
            #     comp = comp.reshape(1, 136)
            trans.pychubby(inp_path_n, face_kp_n, scale, comp[0, :68], comp[0, 68:], outdir+"{:.2f}".format(var_arr[i,1])+"_"+str(i)+".jpg")
               # trans.pychubby(inp_path_n, face_kp_n, 1, comp[0, :68], comp[0, 68:], outdir1+str(j)+".jpg")
            # h.img_folder_to_video_writer(outdir1, outdir1[:-1]+".avi", 30)
            # print("Video Writing Done")

    def vis2level(self, X_train, U_best, U_best_components, V_best_T, outdir, U_best_transform=None, var_dir=None, alpha=100):
        # inp_path_n = "/mnt/scratch2/csy207576/project/AU_Unsupervised/alphapose/alpha_in/000_n.jpg"
        # inp_path_n = "../alphapose/alpha_in/000_n_scaled.jpg"
        trans = Transform()
        h = Helper()

        var_arr, _, _ = h.var_and_cumvar(X_train, U_best_components@V_best_T, U_best_transform, var_dir)
        # var_arr, _, _ = h.var_and_cumvar(U_best, U_best_components, U_best_transform, var_dir)

        
        for i in range(U_best_components.shape[0]):
            comp = np.zeros((1, 136))
            for comp_no in range(U_best_components.shape[1]):
                comp += U_best_components[i, comp_no]*V_best_T[comp_no, :]
            # comp = comp[:, :136] - comp[:, 136:]
            nm = np.linalg.norm(comp)
            if nm!=0:
                comp = comp/nm
            trans.pychubby(inp_path_n, face_kp_n, alpha, comp[0, :68], comp[0, 68:], outdir+"{:.2f}".format(var_arr[i,1])+"_"+str(i)+".jpg")






    def vis_video(self, X, face_kp_n, loadings, outdir,face_part, X_transform=None, var_dir=None, alpha=75):
        # rows of loadings are normalized components
        # inp_path_n = "/mnt/scratch2/csy207576/project/AU_Unsupervised/alphapose/alpha_in/000_n.jpg"
        a = face_part[0]
        b = face_part[1]
        inp_path_n = "../alphapose/alpha_in/000_n_scaled.jpg"
        trans = Transform()
        h = Helper()
        var_arr, _, _ = h.var_and_cumvar(X, loadings, X_transform, var_dir)

        for i in range(0, loadings.shape[0]):
            comp = np.zeros(136)
            comp[a:b] = loadings[i, :(b-a)]
            comp[(a+68):(b+68)] = loadings[i, (b-a):]
            comp = comp.reshape(1, 136)
            out_path = outdir+"{:.2f}".format(var_arr[i,1])+"/"
            try:
                os.makedirs(out_path)
                a = 0
                for s in [s/2 for s in range(0, alpha*2, 1)]:
                    trans.pychubby(inp_path_n, face_kp_n, s, comp[0, :68], comp[0, 68:], out_path+str(a)+".jpg")
                    a+=1
                h.img_folder_to_video_writer(out_path,  out_path+"{:.2f}".format(var_arr[i,1])+".avi", 60)
            except Exception as e:
                print(str(e))   
                h.img_folder_to_video_writer(out_path,  out_path+"{:.2f}".format(var_arr[i,1])+".avi", 60)


    # def comp_similarity_across_datasets(self, main_data, other_data_list, method="L2"):








class Metrics:

    def __init__(self, train_data=None, test_data=None, part=None):
        self.part = part
        self.train_data=train_data
        self.test_data = test_data
        self.var_explained_ = None
        self.avg_var_explained_ = None
        self.avg_row_nonzero_ = None

    def sparsity(self, X):
        total = 1
        for dim in X.shape:
            total = total*dim
        return 100*(1 - np.count_nonzero(X)/float(total))

    def var_explained(self, X, X_approx):

        if self.part=='full_face':
            if "DISFA" in self.train_data or "DISFA" in self.test_data:
                X[:, 60], X[:, 64] = 0, 0
                X[:, 60+68], X[:, 64+68] = 0, 0
                X_approx[:, 60], X_approx[:, 64] = 0, 0
                X_approx[:, 60+68], X_approx[:, 64+68] = 0, 0

            if "bp4d" in self.train_data or "bp4d" in self.test_data:
                X[:, 60], X[:, 64], X[:, :17] = 0, 0, 0
                X[:, 60+68], X[:, 64+68], X[:, 68:(68+17)] = 0, 0, 0
                X_approx[:, 60], X_approx[:, 64], X_approx[:, :17] = 0, 0, 0
                X_approx[:, 60+68], X_approx[:, 64+68], X_approx[:, 68:(68+17)] = 0, 0, 0


        if self.part=='lips':
            if "DISFA" in self.train_data or "DISFA" in self.test_data or "bp4d" in self.train_data or "bp4d" in self.test_data:
                X[:, 12], X[:, 16] = 0, 0
                X[:, 12+20], X[:, 16+20] = 0, 0 
                X_approx[:, 12], X_approx[:, 16] = 0, 0
                X_approx[:, 12+20], X_approx[:, 16+20] = 0, 0 

        if self.part=='jawline':
            if "bp4d" in self.train_data or "bp4d" in self.test_data:
                X[:, :], X_approx[:, :] = 0, 0  


        # if self.data_name is not None:
        #     if self.part=='full_face':
        #         if "DISFA" in self.data_name:
        #             X_approx[:, 60], X_approx[:, 64] = 0, 0
        #             X_approx[:, 60+68], X_approx[:, 64+68] = 0, 0
        #         if "bp4d" in self.data_name:
        #             X_approx[:, 60], X_approx[:, 64], X_approx[:, :17] = 0, 0, 0
        #             X_approx[:, 60+68], X_approx[:, 64+68], X_approx[:, 68:(68+17)] = 0, 0, 0

        #     elif self.part=='lips':
        #         if "DISFA" in self.data_name or "bp4d" in self.data_name:
        #             X_approx[:, 12], X_approx[:, 16] = 0, 0
        #             X_approx[:, 12+20], X_approx[:, 16+20] = 0, 0                
        #     elif self.part=='jawline' and "bp4d" in self.data_name:
        #         X_approx[:, :] = 0


        # if self.data_name is not None:
        #     if self.part=='full_face':
        #         if "DISFA" in self.data_name:
        #             X[:, 60], X[:, 64] = 0, 0
        #             X[:, 60+68], X[:, 64+68] = 0, 0
        #         if "bp4d" in self.data_name:
        #             X[:, 60], X[:, 64], X[:, :17] = 0, 0, 0
        #             X[:, 60+68], X[:, 64+68], X[:, 68:(68+17)] = 0, 0, 0

        #     elif self.part=='lips':
        #         if "DISFA" in self.data_name or "bp4d" in self.data_name:
        #             X[:, 12], X[:, 16] = 0, 0
        #             X[:, 12+20], X[:, 16+20] = 0, 0                
        #     elif self.part=='jawline' and "bp4d" in self.data_name:
        #         X[:, :] = 0

        
        # X and X_approx need to be numpy array
        ve = 100 - 100*(np.linalg.norm(X - X_approx)**2)/(np.linalg.norm(X)**2)
        self.var_explained_ = ve    
        return ve

    def avg_var_explained(self, X, X_approx):
        ave = 100 - 100*np.mean(np.sum((X - X_approx) ** 2, axis=1) / np.sum(X ** 2, axis=1))
        self.avg_var_explained_ = ave
        return ave

    def avg_row_nonzero(self, X):
        arn = np.mean(np.count_nonzero(X, axis=1))
        self.avg_row_nonzero_ = arn
        return arn


    def symmetry_score(self, V):
        # V has a shape n_componentsx136
        # first 68 are x, then remaining are y
        part_list = {'eyebrows':[[17,18,19,20,21],[26,25,24,23,22]], 'eyes':[[36,37,38,39,40,41], [45,44,43,42,47,46]], 'nose':[[31,32],[35,34]], 'lips':[[48,49,50,60,61,67,59,58], [54,53,52,64,63,65,55,56]], 'jawline':[[0,1,2,3,4,5,6,7],[16,15,14,13,12,11,10,9]]}
        sym_score_x, sym_score_y = 0, 0
        corr_x, corr_y = [], []
        # V[V>0] =1 
        # V[V<0] = -1
        for i in range(V.shape[0]):
            x1, x2, y1, y2 = [], [], [], []
            if np.linalg.norm(V[i, :])!=1:
                V[i, :] = V[i, :]/np.linalg.norm(V[i, :])
    
            for part in part_list:
                # print(part)
                for kp1, kp2 in zip(part_list[part][0], part_list[part][1]):
                    # print(i, kp1, kp2)
                    x1.append(V[i, kp1])
                    x2.append(-V[i, kp2])
                    y1.append(V[i, kp1+68])
                    y2.append(V[i, kp2+68])
                    sym_score_x += V[i, kp1]*(-V[i, kp2])
                    sym_score_y += V[i, kp1+68]*(V[i, kp2+68])
            
            corr_x.append(np.corrcoef(x1,x2))
            corr_y.append(np.corrcoef(y1,y2))

        return sym_score_x/V.shape[0], sym_score_y/V.shape[0]
        # return sum(np.array(corr_x)[0,1])/V.shape[0], sum(np.array(corr_y)[0,1])/V.shape[0]


    # def __init__(self, inp_data_path=None):
    #     if inp_data_path is None: self.data = None
    #     else: self.data = pd.read_csv(inp_data_path, index_col=0)


class DictionaryLearner:

# Data format will be the original data format same for all - CK+, DISFA, etc.
    def __init__(self, df_X, input_file_path=None, face_part='left_eyebrow', var_threshold=95, model_prefix=None, model_dir=None, model_path=None):
        # df_X refers to pandas dataframe form of input data
        # X refers to numpy array form of input data
        # 'n' is appended to any variable name at end which is registered   
        self.df_X = df_X
        self.face_part = face_part
        self.var_threshold_two_level = 99
        self.full_var_threshold_two_level = 95
        self.parts = {'left_eyebrow':[17,22], 'right_eyebrow':[22,27], 'left_eye':[36,42], 'right_eye':[42,48], 'jawline':[0,17], 'nose':[27,36], 'lips':[48, 68]}

        if face_part=='full_face':
            self.parts = {'left_eyebrow':[17,22], 'right_eyebrow':[22,27], 'left_eye':[36,42], 'right_eye':[42,48], 'jawline':[0,17], 'nose':[27,36], 'lips':[48, 68], 'full_face':[0, 68]}

        self.face_keypoints = self.parts[face_part]
        self.var_threshold=var_threshold
        self.model_prefix = model_prefix
        self.model_dir = model_dir
        self.model_path = model_path

        self.d = DataLoader()
        print("Shape of df_X : ", df_X.shape)
        self.df_X_delta, cols_kp, cols_kp_n = self.d.delta_loader(df_data=self.df_X)

        self.a = self.face_keypoints[0]
        self.b = self.face_keypoints[1]
        cols_kp_n = cols_kp_n[0]+cols_kp_n[1]
        self.cols_part_n = list(cols_kp_n[self.a:self.b]) + list(cols_kp_n[(self.a+68):(self.b+68)])
        # print(len(cols_kp_n), cols_part_n)
        self.df_X_only_delta = self.df_X_delta[self.cols_part_n]
        # print(self.df_X_only_delta.shape)
        
    def train(self, X_train, X_test, train_data=None, test_data=None, sub_id=None, video_id=None, training=True, alpha=None, val=None):
        # X_train and X_test to be provided as numpy arrays
        # print("Value of training: ",training)
        print("Training in train function is : ", training)
        h = Helper(self.face_part)
        m_train = Metrics()
        m_test = Metrics(train_data, test_data, self.face_part)
        # print("Shape of X_train : ", X_train.shape)

        if training == True:
            print("Training the model ...")
            flag = 0

            # if val is None:
            #     val_list = [i for i in range(10, X_train.shape[1]+1)]
            # else:
            #     val_list = [i for i in range(val, 0, -1)]

            if val is None:
                val_list = [i for i in range(1, X_train.shape[1]+1)]
            else:
                val_list = [i for i in range(1, val+1)]
                # val_list = [val]


            if alpha is None:
                alpha_list = np.arange(10, 0, -0.5)
            else:
                alpha_list = [i for i in np.arange(alpha, 0, -0.5)]

            for val in val_list:
                for alpha in alpha_list: 
            # for val in [5]:
            #     for alpha in [1]:
                    # val = 1.0*val
                    # val has to be integer since it represents the number of components
                    print("val is : ", val, "alpha is : ", alpha)
                    alpha = 1.0*alpha
                    dict_learner = DictionaryLearning(n_components=val, alpha=alpha, max_iter=2000, fit_algorithm='cd', transform_algorithm='lasso_cd', transform_alpha=alpha, random_state=42, positive_code=True, transform_max_iter=2000)
                    dict_learner.fit(X_train)
                    m_train.var_explained_ = m_train.var_explained(X_train, dict_learner.transform(X_train)@dict_learner.components_)
                    print("val : ", val, "alpha : ", alpha, "Var Explained on Train : ", m_train.var_explained_)
                    if m_train.var_explained_<self.var_threshold:
                        continue
                    else:
                        if self.model_dir is None:
                            if sub_id is not None:
                                self.model_dir = self.model_prefix+sub_id+"/"+str(val)+"_comp_"+str(alpha)+"_alpha_"+"cd"+"_fa_"+"lasso_cd"+"_ta_"+"True"+"_pc_"+str(2000)+"2000_"+str(2000)+"_tmi/"
                            elif video_id is not None:
                                self.model_dir = self.model_prefix+video_id+"/"+str(val)+"_comp_"+str(alpha)+"_alpha_"+"cd"+"_fa_"+"lasso_cd"+"_ta_"+"True"+"_pc_"+str(2000)+"_mi_"+str(2000)+"_tmi/"                        
                            else:
                                self.model_dir = self.model_prefix+str(val)+"_comp_"+str(alpha)+"_alpha_"+"cd"+"_fa_"+"lasso_cd"+"_ta_"+"True"+"_pc_"+str(2000)+"_mi_"+str(2000)+"_tmi/"
                        
                        try:
                            os.makedirs(self.model_dir)
                        except Exception as e:
                            print(str(e))
                            pass
                        print(self.model_prefix)
                        if self.model_path is None:
                            self.model_path = self.model_dir + "dict_learner_"+str(val)+"_comp_"+str(alpha)+"_alpha_"+"lasso_cd"+"_ta.pkl"
                        flag = 1
                        print(self.model_path)
                        pickle.dump(dict_learner, open(self.model_path, 'wb'))
                        m_test.var_explained_ = m_test.var_explained(X_test, dict_learner.transform(X_test)@dict_learner.components_)
                        m_train.avg_var_explained_ = m_train.avg_var_explained(X_train, dict_learner.transform(X_train)@dict_learner.components_)
                        m_test.avg_var_explained_ = m_test.avg_var_explained(X_test, dict_learner.transform(X_test)@dict_learner.components_)
                                  
                        print("Variance Explained on the Test : ", m_test.var_explained_)
                        X_train_transform = dict_learner.transform(X_train)
                        pd.DataFrame(X_train_transform).to_csv(self.model_dir+"X_train_transform.csv")
                        np.savetxt(self.model_dir+"X_train_transform_avg_row_nonzero.txt", np.array([m_train.avg_row_nonzero(X_train_transform), 0]))     
                        pd.DataFrame(dict_learner.components_).to_csv(self.model_dir+"loadings.csv")

                        break
                if flag==1:
                    break

        else: 
            if self.model_dir is None:
                if sub_id is not None:
                    self.model_dir = h.find_minvar_DL_path(self.model_prefix+sub_id+"/", self.var_threshold)
                    # for model_dir in glob.glob(self.model_prefix+sub_id+"/*tmi/", recursive=True):
                    #     if pd.read_csv(model_dir+"train_var.csv", index_col=0).iloc[0,0]>=95:
                    #         self.model_dir = model_dir
                    #         break
                elif video_id is not None:
                    self.model_dir = h.find_minvar_DL_path(self.model_prefix+video_id+"/", self.var_threshold)                    
                    # for model_dir in glob.glob(self.model_prefix+video_id+"/*tmi/", recursive=True):
                    #     if pd.read_csv(model_dir+"train_var.csv", index_col=0).iloc[0,0]>=95:
                    #         self.model_dir = model_dir
                    #         break
                else:
                    self.model_dir = h.find_minvar_DL_path(self.model_prefix, self.var_threshold)                    
                    # for model_dir in glob.glob(self.model_prefix+"*tmi/", recursive=True):
                    #     if pd.read_csv(model_dir+"train_var.csv", index_col=0).iloc[0,0]>=95:
                    #         self.model_dir = model_dir
                    #         break

            print("model dir is : ", self.model_dir)
            self.model_path = glob.glob(self.model_dir+"*.pkl", recursive=True)[0]

            dict_learner = pickle.load(open(self.model_path, 'rb'))
            m_train.var_explained_ = m_train.var_explained(X_train, dict_learner.transform(X_train)@dict_learner.components_)
            m_test.var_explained_ = m_test.var_explained(X_test, dict_learner.transform(X_test)@dict_learner.components_)

            X_train_transform = pd.read_csv(self.model_dir+"X_train_transform.csv", index_col=0).to_numpy()
            np.savetxt(self.model_dir+"X_train_transform_avg_row_nonzero.txt", np.array([m_train.avg_row_nonzero(X_train_transform), 0])) 

        h.var_and_cumvar(X_train, dict_learner.components_, dict_learner.transform(X_train), self.model_dir)
        try:
            os.makedirs(self.model_dir+"vis/")
        except Exception as e:
            print(str(e))
        h.vis(X_train, dict_learner.components_, dict_learner.components_, self.model_dir+"vis/", [self.a, self.b], X_train_transform, self.model_dir, 100)

        pca_m_train = Metrics()
        pca_m_test = Metrics(train_data, test_data, self.face_part)
        for val in range(1, X_train.shape[1]):
            pca = PCA(n_components=val)
            pca.fit(X_train)
            pca_X_train_transform = X_train@(pca.components_.T)
            pca_m_train.var_explained_ = pca_m_train.var_explained(X_train, pca_X_train_transform@pca.components_)
            if pca_m_train.var_explained_>=self.var_threshold:
                break

        try:
            os.makedirs(self.model_dir+"pca_vis/")
        except Exception as e:
            print(str(e))
        h.vis(X_train, pca.components_, pca.components_, self.model_dir+"pca_vis/", [self.a, self.b], pca_X_train_transform, self.model_dir+"pca_", 100)



        pca_X_train_transform = X_train@(pca.components_.T) 

        fp_dict = {'left_eyebrow':'L_eyebrow', 'right_eyebrow':'R_eyebrow', 'left_eye':'L_eye', 'right_eye':'R_eye', 'jawline':'jawline', 'nose':'nose', 'lips':'lips'}
        fp = fp_dict[self.face_part]

        # h.plot_hist(X_train_transform, pca_X_train_transform, self.model_dir+"train_transform_hist.png", 'PFM('+fp+')')

        np.savetxt(self.model_dir+"X_train_transform_avg_row_nonzero.txt", np.array([m_train.avg_row_nonzero(X_train_transform), 0]))         
        np.savetxt(self.model_dir+"X_train_var_explained.txt", np.array([m_train.var_explained_, 0]))         
  
        np.savetxt(self.model_dir+"pca_X_train_transform_avg_row_nonzero.txt", np.array([pca_m_train.avg_row_nonzero(pca_X_train_transform), 0]))         
        np.savetxt(self.model_dir+"pca_X_train_var_explained.txt", np.array([pca_m_train.var_explained_, 0]))         
        np.savetxt(self.model_dir+"pca_n_components.txt", np.array([pca.components_.shape[0], 0]))         
            

        pd.DataFrame(pca_X_train_transform).to_csv(self.model_dir+"pca_X_train_transform.csv")     
        pd.DataFrame(pca.components_).to_csv(self.model_dir+"pca_loadings.csv")     

        return [dict_learner, m_train, m_test, pca, pca_m_train, pca_m_test]





    def random_split_train(self, split_ratio=0.8, training=True):
        df_X_only_delta = self.df_X_only_delta.sample(frac=1)
        X_delta = df_X_only_delta.to_numpy()
        split_point = int(split_ratio*X_delta.shape[0])
        X_train, X_test = X_delta[:split_point,:], X_delta[split_point:, :]
        trained_model = self.train(X_train, X_test, training=training)
        pd.DataFrame([trained_model[1].var_explained_]).to_csv(self.model_dir+"train_var.csv")
        pd.DataFrame([trained_model[2].var_explained_]).to_csv(self.model_dir+"test_var.csv")
        pd.DataFrame([trained_model[1].avg_var_explained_]).to_csv(self.model_dir+"avg_train_var.csv")
        pd.DataFrame([trained_model[2].avg_var_explained_]).to_csv(self.model_dir+"avg_test_var.csv")
        return trained_model



    def leave_video_out_train(self, training=True, unique_sub_vid_pair_list=None):
        if unique_sub_vid_pair_list is None:
            unique_sub_vid_pair_list = self.df_X_delta.groupby(['Subject_id','Video_id']).size().reset_index().rename(columns={0:'count'})[['Subject_id', 'Video_id']]
        model_list = []
        var_explained_list = []
        report = []
        avg_var_explained = 0
        # avg_var_explained_train = []
        # avg_var_explained_test = []
        h = Helper()

        def parallel_run(sub_id, vid_id, return_dict):
            mask =(self.df_X_delta["Subject_id"]!=sub_id) & (self.df_X_delta["Video_id"]!=vid_id)
            # print(mask)
            X_train, X_test = self.df_X_delta[mask][self.cols_part_n], self.df_X_delta[~mask][self.cols_part_n]
            self.model_dir, self.model_path = None, None #Making them None so that new directories will be created for each subject
            sub_id = str(sub_id)
            vid_id = str(vid_id)

            temp_path = h.find_minvar_DL_path("/".join(self.model_prefix.split("/")[:-2])+"/inter_data/", self.var_threshold)                    
            val = temp_path.split("/")[-2].split("_")[0]
            alpha = temp_path.split("/")[-2].split("_")[2]
            # print("val and alpha is :", val, alpha)
            val = int(val)
            alpha = float(alpha)

            trained_model = self.train(X_train, X_test, sub_id=sub_id+"_"+vid_id, training=training, alpha=alpha, val=val)
            model_list.append(trained_model[0])
            var_explained_list.append([trained_model[1], trained_model[2]])
            report.append([sub_id, vid_id, trained_model[1].var_explained_, trained_model[2].var_explained_])
            # avg_var_explained_train.append(trained_model[1].var_explained_)  
            # avg_var_explained_test.append(trained_model[2].var_explained_)
            pd.DataFrame([trained_model[1].var_explained_]).to_csv(self.model_dir+"train_var.csv")
            pd.DataFrame([trained_model[2].var_explained_]).to_csv(self.model_dir+"test_var.csv")
            return_dict[sub_id+'_'+vid_id] = [trained_model[1].var_explained_, trained_model[2].var_explained_]


        # manager = mp.Manager()
        # return_dict = manager.dict()
        # jobs = []
        # for sub_id, vid_id in zip(unique_sub_vid_pair_list.iloc[:,0], unique_sub_vid_pair_list.iloc[:,1]):
        #     p = mp.Process(target=parallel_run, args=(sub_id, vid_id, return_dict))
        #     p.start()
        #     jobs.append(p)

        # for p in jobs:
        #     p.join()

        max_processes = 10
        max_process_list = [a for a in range(0, unique_sub_vid_pair_list.shape[0], max_processes)]
        max_process_list.append(max_process_list[-1]+unique_sub_vid_pair_list.shape[0]%max_processes)

        manager = mp.Manager()
        return_dict = manager.dict()
        for i in range(len(max_process_list[:-1])):
            # 40, 80, 120, .... will be the increments
            jobs = []
            for ind in range(max_process_list[i], max_process_list[i+1]):
                sub_id = unique_sub_vid_pair_list.iloc[ind, 0]
                vid_id = unique_sub_vid_pair_list.iloc[ind, 1]
                print(sub_id, vid_id)
                # exit()
                p = mp.Process(target=parallel_run, args=(sub_id, vid_id, return_dict))
                p.start()
                jobs.append(p)

            for p in jobs:
                p.join()


        avg_var_explained_train = [val[0] for val in return_dict.values()]
        avg_var_explained_test = [val[1] for val in return_dict.values()]
        avg_var_explained_train = sum(avg_var_explained_train)/unique_sub_vid_pair_list.shape[0]
        avg_var_explained_test = sum(avg_var_explained_test)/unique_sub_vid_pair_list.shape[0]

        pd.DataFrame(report, columns=["Subject_id", "Video_id", "var_explained","Leave_out_var_explained"]).to_csv(self.model_prefix+"report.csv")
        pd.DataFrame([avg_var_explained_test]).to_csv(self.model_prefix+"avg_variance_explained.csv")
                
        return unique_sub_vid_pair_list, model_list, var_explained_list, avg_var_explained_train, avg_var_explained_test



    def leave_subject_out_train(self, training=True):
        h = Helper()
        sub_id_list = sorted(list(set(self.df_X_delta["Subject_id"])))
        # sub_id_list = ['S080']
        model_list = []
        var_explained_list = []
        report = []
        avg_var_explained = 0
        # avg_var_explained_train = []
        # avg_var_explained_test = []


        def parallel_run(sub_id, return_dict):
            mask = self.df_X_delta["Subject_id"]!=sub_id
            X_train, X_test = self.df_X_delta[mask][self.cols_part_n], self.df_X_delta[~mask][self.cols_part_n]
            self.model_dir, self.model_path = None, None #Making them None so that new directories will be created for each subject            
            temp_path = h.find_minvar_DL_path("/".join(self.model_prefix.split("/")[:-2])+"/inter_data/", self.var_threshold)                    
            val = temp_path.split("/")[-2].split("_")[0]
            alpha = temp_path.split("/")[-2].split("_")[2]
            # print("val and alpha is :", val, alpha)
            val = int(val)
            alpha = float(alpha)
            trained_model = self.train(X_train, X_test, sub_id=sub_id, training=training, alpha=alpha, val=val)
            model_list.append(trained_model[0])
            var_explained_list.append([trained_model[1], trained_model[2]])
            report.append([sub_id, trained_model[1].var_explained_, trained_model[2].var_explained_])
            # print("report is : ", report)
            # avg_var_explained_train.append(trained_model[1].var_explained_)  
            # avg_var_explained_test.append(trained_model[2].var_explained_)
            # above 2 lines won't work as expected in multiprocessing, updating a global variable in multiprocessing function calls
            # just won't update it, check by doing a+=1 by defining a outside function, a won't be updated in multiprocessing calls
            pd.DataFrame([trained_model[1].var_explained_]).to_csv(self.model_dir+"train_var.csv")
            pd.DataFrame([trained_model[2].var_explained_]).to_csv(self.model_dir+"test_var.csv")
            return_dict[sub_id] = [trained_model[1].var_explained_, trained_model[2].var_explained_]

        print(len(sub_id_list))
        max_process_list = [a for a in range(0, len(sub_id_list), 40)]
        max_process_list.append(max_process_list[-1]+len(sub_id_list)%40)
        # print(max_process_list)
        # exit()
        manager = mp.Manager() # manager is used to get return values in multiprocessing calls
        return_dict = manager.dict()
        for i in range(len(max_process_list[:-1])):
            # 40, 80, 120, .... will be the increments
            jobs = []
            # pool = mp.Pool()
            for sub_id in sub_id_list[max_process_list[i]:max_process_list[i+1]]:
                p = mp.Process(target=parallel_run, args=(sub_id, return_dict))
            #    pool.apply_async(parallel_run, args=(sub_id, ))
            # pool.close()
            # pool.join()
                jobs.append(p)
                p.start()

            for p in jobs:
                p.join()

        # print(return_dict.values())
        avg_var_explained_train = [val[0] for val in return_dict.values()]
        avg_var_explained_test = [val[1] for val in return_dict.values()]
        avg_var_explained_train = sum(avg_var_explained_train)/len(sub_id_list)
        avg_var_explained_test = sum(avg_var_explained_test)/len(sub_id_list)

        pd.DataFrame(report, columns=["Subject_id", "var_explained","Leave_out_var_explained"]).to_csv(self.model_prefix+"report.csv")
        pd.DataFrame([avg_var_explained_test]).to_csv(self.model_prefix+"avg_variance_explained.csv")
                
        return sub_id_list, model_list, var_explained_list, avg_var_explained_train, avg_var_explained_test


    def inter_dataset_train(self, df_X_train, df_X_test, train_data_name, test_data_name, training=True):
        df_X_train_delta, _, cols_n = self.d.delta_loader(df_data=df_X_train)
        df_X_test_delta, _, cols_n = self.d.delta_loader(df_data=df_X_test)

        X_train = df_X_train_delta[self.cols_part_n].to_numpy()
        X_test = df_X_test_delta[self.cols_part_n].to_numpy()
        trained_model = self.train(X_train, X_test, train_data_name, test_data_name, training=training)
        
        try:
            os.makedirs(self.model_dir+test_data_name+"/")
        except Exception as e:
            print(str(e))

        pd.DataFrame([trained_model[2].var_explained_]).to_csv(self.model_dir+test_data_name+"/"+"test_var.csv")
        pd.DataFrame([trained_model[1].var_explained_]).to_csv(self.model_dir+"train_var.csv")

        X_test_transform = trained_model[0].transform(X_test)
        m_test = trained_model[2]
        np.savetxt(self.model_dir+test_data_name+"/X_test_transform_avg_row_nonzero.txt", np.array([m_test.avg_row_nonzero(X_test_transform), 0])) 
        pd.DataFrame(X_test_transform).to_csv(self.model_dir+test_data_name+"/"+"X_test_transform.csv")     
        
        pca = trained_model[3]
        pca_m_test = trained_model[5]
        pca_X_test_transform = X_test@(pca.components_.T)       
        pca_m_test.var_explained_ = pca_m_test.var_explained(X_test, pca_X_test_transform@pca.components_)
        
        np.savetxt(self.model_dir+test_data_name+"/pca_X_test_transform_avg_row_nonzero.txt", np.array([pca_m_test.avg_row_nonzero(pca_X_test_transform), 0]))         
        np.savetxt(self.model_dir+test_data_name+"/pca_X_test_var_explained.txt", np.array([pca_m_test.var_explained_, 0]))         
  
        pd.DataFrame(pca_X_test_transform).to_csv(self.model_dir+test_data_name+"/pca_X_test_transform.csv")

        return trained_model



    def two_level_train_nmf(self, X_train, X_test, U_train, V_train, U_test, pca_U_train, pca_V_train, pca_U_test, train_data, test_data, training, l1r=1):
        h = Helper()
        m_test_full = Metrics(train_data, test_data, 'full_face')
        m_train_full = Metrics()
        m_train_1 = Metrics()
        m_test_1 = Metrics(train_data, test_data, 'full_face')
        m_train_2 = Metrics()
        m_test_2 = Metrics()
        m = Metrics()
        pvt = self.var_threshold
        mi = 1000
        print("training value inside 2 level nmf:", training)

        if training is True:
            print("Training the model ...")
            flag = 0
            # for val in range(1, U.shape[1]+1):
            #     for alpha_W in np.arange(5, 0, -0.5): 
            #         for alpha_H in np.arange(5, 0, -0.5):
            # for val, alpha_W, alpha_H in itertools.product(range(5, U_train.shape[1]+1), [0,0.00001], [0,0.00001]):
            for val, alpha_W, alpha_H in itertools.product(26, [10], [10]):
                alpha_W = 1.0*alpha_W
                alpha_H = 1.0*alpha_H
                nmf = NMF(n_components=val, init='random', max_iter=mi, random_state=0, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1r)
                nmf.fit(U_train)
                m_train_full.var_explained_ = m_train_full.var_explained(X_train, (nmf.transform(U_train)@nmf.components_)@V_train)
                print(" val: ",val, " alpha_H: ", alpha_H, " alpha_W: ", alpha_W)
                m_train_2.var_explained_ = m_train_2.var_explained(U_train, nmf.transform(U_train)@nmf.components_)
                print("Variance explained at level 2 :", m_train_2.var_explained_)
                print("FUll variance:", m_train_full.var_explained_)
                if m_train_full.var_explained_<self.full_var_threshold_two_level:
                    continue
                else:
                    if self.model_dir is None:
                        self.model_dir = self.model_prefix+str(val)+"_val_"+str(alpha_H)+"_alpha_H_"+str(alpha_W)+"_alpha_W/" 
                    try:
                        os.makedirs(self.model_dir)
                    except Exception as e:
                        print(str(e))
                        pass
                    if self.model_path is None:
                        self.model_path = self.model_dir+"nmf_"+str(val)+"_comp_"+str(alpha_W)+"_alpha_W_"+str(alpha_H)+"_alpha_H_"+str(l1r)+"_l1r_"+str(pvt)+"_pvt.pkl"
                    flag = 1
                    print(self.model_path)
                    pickle.dump(nmf, open(self.model_path, 'wb'))
                    break

        else:
            if self.model_dir is None:
                for model_dir in glob.glob(self.model_prefix+"*alpha_W/", recursive=True):
                    if pd.read_csv(model_dir+"full_train_var.csv", index_col=0).iloc[0,0]>=self.full_var_threshold_two_level:
                        self.model_dir = model_dir
                        break

            print("model dir is : ", self.model_dir)
            self.model_path = glob.glob(self.model_dir+"*.pkl", recursive=True)[0]

            nmf = pickle.load(open(self.model_path, 'rb'))


        pca_m_train_1 = Metrics()
        pca_m_test_1 = Metrics(train_data, test_data, 'full_face')
        pca_m_train_2 = Metrics()
        pca_m_test_2 = Metrics()
        pca_m_train_full = Metrics()
        pca_m_test_full = Metrics(train_data, test_data, 'full_face')
        for val in range(1, U_train.shape[1]):
            pca = PCA(n_components=val)
            pca.fit(U_train)
            pca_U_train_transform = U_train@(pca.components_.T)
            pca_m_train_full.var_explained_ = pca_m_train_full.var_explained(X_train, pca_U_train_transform@pca.components_@V_train)
            if pca_m_train_full.var_explained_>=self.full_var_threshold_two_level:
                break


        pca_both_m_train_1 = Metrics()
        pca_both_m_test_1 = Metrics(train_data, test_data, 'full_face')
        pca_both_m_train_2 = Metrics()
        pca_both_m_test_2 = Metrics()
        pca_both_m_train_full = Metrics()
        pca_both_m_test_full = Metrics(train_data, test_data, 'full_face')
        for val in range(1, U_train.shape[1]):
            pca_both = PCA(n_components=val)
            pca_both.fit(pca_U_train)
            pca_both_U_train_transform = pca_U_train@(pca_both.components_.T)
            pca_both_m_train_full.var_explained_ = pca_both_m_train_full.var_explained(X_train, pca_both_U_train_transform@pca_both.components_@pca_V_train)
            if pca_both_m_train_full.var_explained_>=self.full_var_threshold_two_level:
                break

        pca_U_train_transform = U_train@(pca.components_.T) 
        pca_U_test_transform = U_test@(pca.components_.T)       

        pca_both_U_train_transform = pca_U_train@(pca_both.components_.T) 
        pca_both_U_test_transform = pca_U_test@(pca_both.components_.T) 
   
        U_train_transform = nmf.transform(U_train)
        U_test_transform = nmf.transform(U_test)

        # h.plot_hist(U_train_transform, pca_U_train_transform, self.model_dir+"U_train_transform_hist.png", "HM")
        
        try:
            os.makedirs(self.model_dir+"vis/")
        except Exception as e:
            print(str(e))

        h.vis2level(X_train, U_train, nmf.components_, V_train, self.model_dir+"vis/", U_train_transform, self.model_dir, alpha=100)
        
        try:
            os.makedirs(self.model_dir+"vis_pca/")
        except Exception as e:
            print(str(e))
        
        h.vis2level(X_train, U_train, pca.components_, V_train, self.model_dir+"vis_pca/", pca_U_train_transform, self.model_dir+"pca_", alpha=100)
        
        try:
            os.makedirs(self.model_dir+"vis_pca_both/")
        except Exception as e:
            print(str(e))
        
        h.vis2level(X_train, pca_U_train, pca_both.components_, pca_V_train, self.model_dir+"vis_pca_both/", pca_both_U_train_transform, self.model_dir+"pca_both_", alpha=100)
        



        print("Sparsity of U_train_components: ", m.sparsity(nmf.components_))
        print("Sparsity of U_train_transform: ", m.sparsity(U_train_transform))
        print("Sparsity of U_test_transform: ", m.sparsity(U_test_transform))
        print("Row non-zero of U_train_transform: ", m.avg_row_nonzero(U_train_transform))
        print("Row non-zero of U_test_transform: ", m.avg_row_nonzero(U_test_transform))

        m_train_1.var_explained_ = m_train_1.var_explained(X_train, U_train@V_train)
        m_test_1.var_explained_ = m_test_1.var_explained(X_test, U_test@V_train)

        pca_m_train_1.var_explained_ = pca_m_train_1.var_explained(X_train, pca_U_train@pca_V_train)
        pca_m_test_1.var_explained_ = pca_m_test_1.var_explained(X_test, pca_U_test@pca_V_train)

        pca_both_m_train_1.var_explained_ = pca_both_m_train_1.var_explained(X_train, pca_U_train@pca_V_train)
        pca_both_m_test_1.var_explained_ = pca_both_m_test_1.var_explained(X_test, pca_U_test@pca_V_train)

        m_train_2.var_explained_ = m_train_2.var_explained(U_train, nmf.transform(U_train)@nmf.components_)
        m_test_2.var_explained_ = m_test_2.var_explained(U_test, nmf.transform(U_test)@nmf.components_)
        # m_train.avg_var_explained_ = m_train.avg_var_explained(X_train, nmf.transform(X_train)@nmf.components_)
        # m_test.avg_var_explained_ = m_test.avg_var_explained(X_test, nmf.transform(X_test)@nmf.components_)

        pca_m_train_2.var_explained_ = pca_m_train_2.var_explained(U_train, pca_U_train_transform@pca.components_)
        pca_m_test_2.var_explained_ = pca_m_test_2.var_explained(U_test, pca_U_test_transform@pca.components_)

        pca_both_m_train_2.var_explained_ = pca_both_m_train_2.var_explained(pca_U_train, pca_both_U_train_transform@pca_both.components_)
        pca_both_m_test_2.var_explained_ = pca_both_m_test_2.var_explained(pca_U_test, pca_both_U_test_transform@pca_both.components_)

        m_train_full.var_explained_ = m_train_full.var_explained(X_train, (U_train_transform@nmf.components_)@V_train)
        m_test_full.var_explained_ = m_test_full.var_explained(X_test, (U_test_transform@nmf.components_)@V_train)
        
        pca_m_train_full.var_explained_ = pca_m_train_full.var_explained(X_train, (pca_U_train_transform@pca.components_)@V_train)
        pca_m_test_full.var_explained_ = pca_m_test_full.var_explained(X_test, (pca_U_test_transform@pca.components_)@V_train)

        pca_both_m_train_full.var_explained_ = pca_both_m_train_full.var_explained(X_train, (pca_both_U_train_transform@pca_both.components_)@pca_V_train)
        pca_both_m_test_full.var_explained_ = pca_both_m_test_full.var_explained(X_test, (pca_both_U_test_transform@pca_both.components_)@pca_V_train)
        
        pd.DataFrame([m_train_full.var_explained_]).to_csv(self.model_dir+"full_train_var.csv")
        pd.DataFrame([m_test_full.var_explained_]).to_csv(self.model_dir+test_data+"/full_test_var.csv")

        pd.DataFrame([pca_m_train_full.var_explained_]).to_csv(self.model_dir+"pca_full_train_var.csv")
        pd.DataFrame([pca_m_test_full.var_explained_]).to_csv(self.model_dir+test_data+"/pca_full_test_var.csv")

        pd.DataFrame([pca_both_m_train_full.var_explained_]).to_csv(self.model_dir+"pca_both_full_train_var.csv")
        pd.DataFrame([pca_both_m_test_full.var_explained_]).to_csv(self.model_dir+test_data+"/pca_both_full_test_var.csv")

        pd.DataFrame(U_train_transform).to_csv(self.model_dir+"U_train_transform.csv")
        np.savetxt(self.model_dir+"U_train_avg_row_nonzero.txt", np.array([m_train_full.avg_row_nonzero(U_train), 0]))         
        np.savetxt(self.model_dir+"pca_U_train_avg_row_nonzero.txt", np.array([pca_m_train_2.avg_row_nonzero(pca_U_train), 0]))  
        np.savetxt(self.model_dir+"U_train_transform_avg_row_nonzero.txt", np.array([m_train_full.avg_row_nonzero(U_train_transform), 0]))         
        np.savetxt(self.model_dir+"pca_U_train_transform_avg_row_nonzero.txt", np.array([pca_m_train_2.avg_row_nonzero(pca_U_train_transform), 0]))         
        np.savetxt(self.model_dir+"pca_U_train_var_explained.txt", np.array([pca_m_train_2.var_explained_, 0]))         
        np.savetxt(self.model_dir+"U_train_var_explained.txt", np.array([m_train_2.var_explained_, 0])) 
        np.savetxt(self.model_dir+"pca_n_components.txt", np.array([pca.components_.shape[0], 0]))         
                              
        np.savetxt(self.model_dir+"face_parts_aggregate_var_explained.txt", np.array([m_train_1.var_explained_, 0])) 
        np.savetxt(self.model_dir+test_data+"/face_parts_aggregate_var_explained.txt", np.array([m_test_1.var_explained_, 0])) 

        np.savetxt(self.model_dir+"pca_face_parts_aggregate_var_explained.txt", np.array([pca_m_train_1.var_explained_, 0])) 
        np.savetxt(self.model_dir+test_data+"/pca_face_parts_aggregate_var_explained.txt", np.array([pca_m_test_1.var_explained_, 0])) 

        np.savetxt(self.model_dir+"pca_face_parts_aggregate_var_explained.txt", np.array([pca.components_.shape[0], 0]))         
            
        pd.DataFrame(nmf.components_).to_csv(self.model_dir+"U_train_loadings.csv")
        pd.DataFrame(pca.components_).to_csv(self.model_dir+"pca_U_train_loadings.csv")                                                                            

        return [nmf, m_train_full, m_test_full, m_train_2, m_test_2, m_train_1, m_test_1, pca, pca_m_train_2, pca_m_test_2, pca_m_train_1, pca_m_test_1]



    def inter_dataset_two_level_train(self, df_X_train, df_X_test, train_data_name, test_data_name, data_type, training):
        
        h = Helper()
        # m = Metrics()

        df_X_train_delta, _, cols_n = self.d.delta_loader(df_data=df_X_train)
        df_X_test_delta, _, cols_n = self.d.delta_loader(df_data=df_X_test)

        cols_n = cols_n[0]+cols_n[1]
        X_train = df_X_train_delta[cols_n].to_numpy()
        X_test = df_X_test_delta[cols_n].to_numpy()

        # if "bp4d" in test_data_name or "bp4d" in train_data_name:
        #     # removing the jawline points
        #     jawline_indices = list(range(0, 17))+list(range(68, 17+68))
        #     X_train[:, jawline_indices] = 0
        #     X_test[:, jawline_indices] = 0
        #     del self.parts['jawline']

        for part in self.parts:
            if part=='jawline' and 'bp4d' in train_data_name:
                continue
            model_prefix = "DL/"+train_data_name+"/reg1_6_pts/"+data_type+"/segmented_face_parts/"+part+"/inter_data/"
            print(model_prefix)
            # temp_path = ""
            # temp_var = 0
            # flag = 0
            # for path in glob.glob(model_prefix+"*tmi/", recursive=True):
            #     var = pd.read_csv(path+"train_var.csv", index_col=0).iloc[0,0]
            #     if var>=95:
            #         if flag==0:
            #             temp_var = var
            #             temp_path = path
            #             flag = 1
            #         elif var<temp_var:
            #             temp_var = var
            #             temp_path = path
            # model_dir = temp_path

            model_dir = h.find_minvar_DL_path(model_prefix, self.var_threshold)

            print(model_dir)

            X_train_transform = pd.read_csv(model_dir+"X_train_transform.csv", index_col=0).to_numpy()  
            X_test_transform = pd.read_csv(model_dir+test_data_name+"/"+"X_test_transform.csv", index_col=0).to_numpy()  
            loadings = pd.read_csv(model_dir+"loadings.csv", index_col=0).to_numpy()

            pca_X_train_transform = pd.read_csv(model_dir+"pca_X_train_transform.csv", index_col=0).to_numpy()  
            pca_X_test_transform = pd.read_csv(model_dir+test_data_name+"/"+"pca_X_test_transform.csv", index_col=0).to_numpy()  
            pca_loadings = pd.read_csv(model_dir+"pca_loadings.csv", index_col=0).to_numpy()

            if 'left_eyebrow' in part:

                U_train = X_train_transform.copy()
                U_test = X_test_transform.copy()
                V_train = h.part_to_68_kp(loadings.copy(), self.parts, part)

                pca_U_train = pca_X_train_transform.copy()
                pca_U_test = pca_X_test_transform.copy()
                pca_V_train = h.part_to_68_kp(pca_loadings.copy(), self.parts, part)    

            else:

                U_train = np.concatenate((U_train, X_train_transform), axis=1)
                V_train = np.concatenate((V_train, h.part_to_68_kp(loadings.copy(), self.parts, part)))
                U_test = np.concatenate((U_test, X_test_transform), axis=1)

                pca_U_train = np.concatenate((pca_U_train, pca_X_train_transform), axis=1)
                pca_V_train = np.concatenate((pca_V_train, h.part_to_68_kp(pca_loadings.copy(), self.parts, part)))
                pca_U_test = np.concatenate((pca_U_test, pca_X_test_transform), axis=1)
                             
        # print("Variance at first level: ", m.var_explained(X_train, U_train@V_train))
        trained_model = self.two_level_train_nmf(X_train, X_test, U_train, V_train, U_test, pca_U_train, pca_V_train, pca_U_test, train_data_name, test_data_name, training)

        try:
            os.makedirs(self.model_dir+test_data_name)
        except Exception as e:
            print(str(e))

        pd.DataFrame(U_train).to_csv(self.model_dir+"U_train.csv")
        pd.DataFrame(V_train).to_csv(self.model_dir+"V_train.csv")
        pd.DataFrame([trained_model[2].var_explained_]).to_csv(self.model_dir+test_data_name+"/"+"full_test_var.csv")
        pd.DataFrame([trained_model[1].var_explained_]).to_csv(self.model_dir+"full_train_var.csv")
        U_test_transform = trained_model[0].transform(U_test)
        pca = trained_model[7]
        pca_U_test_transform = U_test@(pca.components_.T)
        m_test_full = trained_model[2]
        pca_m_test_2 = trained_model[9]
        m_test_2 = trained_model[4]

        m_train_1 = trained_model[5]
        m_test_1 = trained_model[6]

        pca_m_train_1 = trained_model[10]
        pca_m_test_1 = trained_model[11]


        np.savetxt(self.model_dir+test_data_name+"/U_test_avg_row_nonzero.txt", np.array([m_test_full.avg_row_nonzero(U_test), 0])) 
        np.savetxt(self.model_dir+test_data_name+"/pca_U_test_avg_row_nonzero.txt", np.array([pca_m_test_2.avg_row_nonzero(pca_U_test), 0]))
        np.savetxt(self.model_dir+test_data_name+"/U_test_transform_avg_row_nonzero.txt", np.array([m_test_full.avg_row_nonzero(U_test_transform), 0])) 
        np.savetxt(self.model_dir+test_data_name+"/pca_U_test_transform_avg_row_nonzero.txt", np.array([pca_m_test_2.avg_row_nonzero(pca_U_test_transform), 0])) 
        np.savetxt(self.model_dir+test_data_name+"/pca_U_test_var_explained.txt", np.array([pca_m_test_2.var_explained_, 0]))         
        np.savetxt(self.model_dir+test_data_name+"/U_test_var_explained.txt", np.array([m_test_2.var_explained_, 0]))         

        np.savetxt(self.model_dir+"level_1_train_var_explained.txt", np.array([m_train_1.var_explained_, 0]))         
        np.savetxt(self.model_dir+test_data_name+"/level_1_test_var_explained.txt", np.array([m_test_1.var_explained_, 0]))         

        np.savetxt(self.model_dir+"pca_level_1_train_var_explained.txt", np.array([pca_m_train_1.var_explained_, 0]))         
        np.savetxt(self.model_dir+test_data_name+"/pca_level_1_test_var_explained.txt", np.array([pca_m_test_1.var_explained_, 0]))         

        pd.DataFrame(U_test_transform).to_csv(self.model_dir+test_data_name+"/"+"U_test_transform.csv")     
        
        return trained_model



    def learn_sparse_code_pure_AUs(self, test_data, val=None, alpha=None):
        print("Learning to sparse code ...")
        AU_list = ["pure_AUs", "sparse_pure_AUs", "AUs_with_combinations"]
        au_dir = "Data/reg1_6_pts/cum_event_front_facing_2level_reg_eyes_eyebrows_jawline_nose/"
        algo_list = ["lasso_lars", "lasso_cd", "threshold"]

        res = []
        for au_type in AU_list:
            print(au_type)
            pure_AUs = pd.read_csv(au_dir+au_type+".csv", index_col=0)
            X = self.df_X_only_delta.to_numpy()
            # X_train = X[:100, :]

            AUs_delta, cols_kp, cols_kp_n = self.d.delta_loader(df_data=pure_AUs)
            cols_kp_n = cols_kp_n[0]+cols_kp_n[1]
            AUs_only_delta = AUs_delta[cols_kp_n].to_numpy()
            print(AUs_only_delta.shape)
            AUs_only_delta = AUs_only_delta/np.linalg.norm(AUs_only_delta, axis=1).reshape(AUs_only_delta.shape[0], -1)
            m = Metrics(None, test_data)
            # X is the data to be sparsified using AUs
            alpha = None
            # if val is None:
            #     val_list = [i for i in range(1, X_train.shape[1]+1)]
            # else:
            #     val_list = [val]
            if alpha is None:
                # alpha_list = np.arange(10, 0, -0.5)
                alpha_list = [0]
            else:
                alpha_list = [i for i in np.arange(alpha, 0, -0.5)]

            # for val in val_list:
            for alpha in alpha_list:
                alpha = 1.0*alpha
                print("alpha : ", alpha)
                for ta in algo_list:
                    print(ta)
                    sparse_coder = SparseCoder(AUs_only_delta, transform_algorithm=ta, transform_alpha=alpha,  positive_code=True, transform_max_iter=2000)
                    X_transform = sparse_coder.transform(X)
                    m.var_explained_ = m.var_explained(X, X_transform@AUs_only_delta)
                    print(m.var_explained_)
                    print("Avg row nonzero is : ", m.avg_row_nonzero(X_transform))
                    res.extend([au_type, ta, m.avg_row_nonzero_, m.var_explained_])
        print(res)
        return res
                        # if m.var_explained_>=95:
                        #     break

        # dict_learner.components_ = AUs_only_delta
        # X_transform = dict_learner.transform(X)
        # print("AUs transform variance is : ", m.var_explained(X, X_transform@AUs_only_delta))

        # X_transform_1 = X@np.linalg.pinv(AUs_only_delta)
        # print("AUs transform variance using least square is : ", m.var_explained(X, X_transform_1@AUs_only_delta))
        # pd.DataFrame(X_transform).to_csv("X_transform_1_for_fun.csv")


    def learn_pseudo_inverse_pure_AUs(self, val=None, alpha=None):

        print("Learning through Pseudo Inverse...")
        AU_list = ["pure_AUs", "sparse_pure_AUs", "AUs_with_combinations"]
        au_dir = "Data/reg1_6_pts/cum_event_front_facing_2level_reg_eyes_eyebrows_jawline_nose/"
        algo_list = ["lasso_lars", "lasso_cd", "threshold"]

        for au_type in AU_list:
            print(au_type)
            pure_AUs = pd.read_csv(au_dir+au_type+".csv", index_col=0)
            X = self.df_X_only_delta.to_numpy()
            # X_train = X[:100, :]

            AUs_delta, cols_kp, cols_kp_n = self.d.delta_loader(df_data=pure_AUs)
            cols_kp_n = cols_kp_n[0]+cols_kp_n[1]
            AUs_only_delta = AUs_delta[cols_kp_n].to_numpy()
            print(AUs_only_delta.shape)
            AUs_only_delta = AUs_only_delta/np.linalg.norm(AUs_only_delta, axis=1).reshape(AUs_only_delta.shape[0], -1)
            m = Metrics()

            X_transform = X@np.linalg.pinv(AUs_only_delta)
            m.var_explained_ = m.var_explained(X, X_transform@AUs_only_delta)
            print(m.var_explained_)
            print("Avg row nonzero is : ", m.avg_row_nonzero(X_transform))


    def learn_least_square_pure_AUs(self, test_data, val=None, alpha=None):

        print("Learning through Least Squares...")
        AU_list = ["pure_AUs", "sparse_pure_AUs", "AUs_with_combinations"]
        au_dir = "Data/reg1_6_pts/cum_event_front_facing_2level_reg_eyes_eyebrows_jawline_nose/"
        algo_list = ["lasso_lars", "lasso_cd", "threshold"]

        res = []
        for au_type in AU_list:
            print(au_type)
            pure_AUs = pd.read_csv(au_dir+au_type+".csv", index_col=0)
            X = self.df_X_only_delta.to_numpy()
            # X_train = X[:100, :]

            AUs_delta, cols_kp, cols_kp_n = self.d.delta_loader(df_data=pure_AUs)
            cols_kp_n = cols_kp_n[0]+cols_kp_n[1]
            AUs_only_delta = AUs_delta[cols_kp_n].to_numpy()
            print(AUs_only_delta.shape)
            AUs_only_delta = AUs_only_delta/np.linalg.norm(AUs_only_delta, axis=1).reshape(AUs_only_delta.shape[0], -1)
            m = Metrics(None, test_data)

            X_transform = np.linalg.lstsq(AUs_only_delta.T, X.T)[0]
            X_transform = np.array(X_transform).T
            m.var_explained_ = m.var_explained(X, X_transform@AUs_only_delta)
            print(m.var_explained_)
            print("Avg row nonzero is : ", m.avg_row_nonzero(X_transform))

            res.extend([au_type, m.avg_row_nonzero_, m.var_explained_])
        print(res)
        return res
