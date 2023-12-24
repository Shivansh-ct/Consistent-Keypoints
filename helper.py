import os
import cv2
import matplotlib
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pychubby.detect import LandmarkFace
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile, AbsoluteMove, Action, Lambda
from skimage.transform import SimilarityTransform, AffineTransform




class DataCreator:

    def delta_creator(self, inp_path, delta_path, delta_only_kp_path):
        dl = DataLoader()
        delta_data, a , b = dl.delta_loader(inp_path)
        cols = a[0]+a[1]+b[0]+b[1]
        delta_data_kp = delta_data[cols]
        delta_data.to_csv(delta_path)
        delta_data_kp.to_csv(delta_only_kp_path)

    # def data_vis(face_kp, inp_path, reg=True, out_path):
    #     d = DataLoader()
    #     X,_ ,X_n ,_  = d.delta_loader(inp_path)
    #     h = Helper()
    #     if reg==True:
    #         h.vis()






class DataLoader:

    def __init__(self):
        self.data = None
        self.delta_data = None
        self.kp_columns = None
        self.kp_n_columns = None

    def data_loader(self, input_file_path):
        self.data = pd.read_csv(input_file_path, index_col=0)
        return self.data
    # , dtype={'Subject_id':str, 'Video_id':str}


    def cart2pol(self, x_arr, y_arr):
        arr = np.zeros((x_arr.shape[0], 2))
        i = 0
        for i in range(arr.shape[0]):
            x = x_arr[i]
            y = y_arr[i]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(abs(y), abs(x))
            if x<0 and y>0:
                theta = np.pi - theta
            if x<0 and y<0:
                theta = np.pi + theta
            if x>0 and y<0:
                theta = 2*np.pi - theta
            arr[i, :] = [r, theta]
        return arr[:, 0], arr[:, 1] 


    def pol2cart(self, r_arr, theta_arr):
        arr = np.zeros((r_arr.shape[0], 2))
        i = 0
        for i in range(arr.shape[0]):
            r = r_arr[i]
            theta = theta_arr[i]
            delta_x = r*np.cos(theta)
            delta_y = r*np.sin(theta)
            # if delta_x<0: print("delta_x: ", delta_x)
            # if delta_y<0: print("delta_y: ", delta_y)
            arr[i, :] = [delta_x, delta_y]
        return arr[:, 0], arr[:, 1] 


    def delta_to_rtheta(self, input_file_path, face_part):
        a = face_part[0]
        b = face_part[1]
        _, _, X, cols = self.delta_loader(input_file_path)
        r_col = []
        theta_col = []
        for i in range(a, b):
            r, theta = self.cart2pol(np.array(X["delta_x"+str(i)+"_Normalized"]), np.array(X["delta_y"+str(i)+"_Normalized"]))
            X["delta_x"+str(i)+"_Normalized"] = r
            X["delta_y"+str(i)+"_Normalized"] = theta
            r_col.append("r"+str(i)+"_Normalized")
            theta_col.append("theta"+str(i)+"_Normalized")
            X.rename(columns={"delta_x"+str(i)+"_Normalized": "r"+str(i)+"_Normalized", "delta_y"+str(i)+"_Normalized": "theta"+str(i)+"_Normalized"}, inplace=True)
        print(r_col, theta_col)
        return X[r_col+theta_col], [r_col, theta_col]


    def rtheta_to_delta(self, X, face_part):
        # X is a dataframe returned by delta_to_rtheta function
        a = face_part[0]
        b = face_part[1]
        x_n_column = []
        y_n_column = []
        for i in range(a, b):
            delta_x, delta_y = self.pol2cart(np.array(X["r"+str(i)+"_Normalized"]), np.array(X["theta"+str(i)+"_Normalized"]))
            X["r"+str(i)+"_Normalized"] = delta_x
            X["theta"+str(i)+"_Normalized"] = delta_y
            col_x = "delta_x"+str(i)+"_Normalized"
            col_y = "delta_y"+str(i)+"_Normalized"
            x_n_column.append(col_x)
            y_n_column.append(col_y)
            X.rename(columns={"r"+str(i)+"_Normalized":col_x, "theta"+str(i)+"_Normalized":col_y}, inplace=True)
        return X[x_n_column+y_n_column], [x_n_column, y_n_column]




    def delta_loader(self, input_file_path=None, df_data=None, n_kp=68, vid=None):
        # input_file_path and df_delta can't be None at the same time
        if input_file_path is not None:
            df_delta = pd.read_csv(input_file_path, index_col=0, dtype={'Subject_id':str, 'Video_id':str})        
        else:
            df_delta = df_data.copy()

        if vid is not None:
            df_delta = df_delta[df_delta["Video_id"]==vid]
        delta_column_list = []
        name_list = ['x', 'y']
        name = ["", "_Normalized"]

        for name2 in name:
            for i, name1 in product(range(n_kp), name_list):
                df_delta["Apex_"+name1+str(i)+name2] = df_delta["Apex_"+name1+str(i)+name2] - df_delta["Offset_"+name1+str(i)+name2]
                df_delta.drop(columns=["Offset_"+name1+str(i)+name2], inplace=True)
                col = "delta_"+name1+str(i)+name2
                delta_column_list.append(col)
                df_delta.rename(columns={"Apex_"+name1+str(i)+name2:col},  inplace=True)
                
                
        counter = 0

        x_column = []
        y_column = []
        x_n_column = []
        y_n_column = []

        for i in delta_column_list:

            if counter <2*n_kp:
                if counter%2==0:
                    x_column.append(i)
                else:
                    y_column.append(i)
            else:
                if counter%2==0:
                    x_n_column.append(i)
                else:
                    y_n_column.append(i)  

            counter+=1
        
        self.delta_data, self.kp_columns, self.kp_n_columns = df_delta, [x_column, y_column], [x_n_column, y_n_column]
            
        # return df_delta[x_column+y_column], [x_column, y_column], df_delta[x_n_column+y_n_column], [x_n_column, y_n_column]     
        return df_delta, [x_column, y_column], [x_n_column, y_n_column]     



    def filter_columns(self, df, column_list):
        return df[column_list]

    def filter_rows(self, df, column, value):
        return df[df[column]==value].copy() 


    def top_k_vis(self, delta_df_only_kp, component, k, inp_path, face_kp, scale_, out_dir):
        arr = np.array(delta_df_only_kp)
        arr = arr/(np.linalg.norm(arr, axis=1).reshape(arr.shape[0], 1))
        corr = arr@(component.reshape(delta_df_only_kp.shape[1], 1))

        top_k_row_indices = []
        t = Transform()
        
        for val in range(k):
            cur_index = np.argmax(corr)
            top_k_row_indices.append([val, cur_index])
            corr[cur_index] = -10000
            print(cur_index)

            delta = arr[cur_index, :]
            try:
                os.makedirs(out_dir)
            except:
                pass

            out_path = out_dir + "/"+"index_"+str(cur_index)+"_"+str(val)+".jpg"
            t.pychubby(inp_path, face_kp, scale_, delta[:68], delta[68:], out_path)
        
        # indices = list(np.array(top_k_row_indices)[:, 1])
        # return delta_df_only_kp.loc[delta_df_only_kp.index[indices]], top_k_row_indices


    # def_top_k_vis(self, delta_df_only_kp, priority_list, out_dir):
    #     # outdir to keep all output visualizations
    #     trans = Transform()














class Transform:
    def __init__(self, inp_path=None, matrix=None):
        self.path = inp_path
        self.matrix = matrix


    def pychubby(self, inp_path, face_kp, K, xdiff, ydiff, out_path, std_dev=None):

        class CustomAction(Action):
            def __init__(self, scale=0.3):
                self.scale = scale
            def perform(self, lf):
                a_l = AbsoluteMove(x_shifts=xdiff_final,y_shifts=ydiff_final)
                return a_l.perform(lf)

        def get_all_au():
            img = cv2.imread(inp_path)
            lf = LandmarkFace.estimate(img)
            lf.points = face_kp
            xdiff_final = dict(enumerate(xdiff*K,start=0))
            ydiff_final = dict(enumerate(ydiff*K,start=0))
            return xdiff_final,ydiff_final,lf

        path = inp_path
        xdiff_final,ydiff_final,lf = get_all_au()
        a_all = CustomAction()
        new_lf, _ = a_all.perform(lf)
    
        thickness = 3
        img = new_lf.img.copy()

        for i in range(68):
            start_point = (int(face_kp[i, 0]), int(face_kp[i,1]))
            end_point = (int(face_kp[i, 0]+xdiff_final[i]),  int(face_kp[i, 1]+ydiff_final[i]))
            dist = np.linalg.norm(np.array(start_point) - np.array(end_point)) + 1e-15
            # if i==48:
            #     print("Keypoint 48 is: ", start_point, end_point, dist)
            tiplength = 5/dist
            img = cv2.circle(img, end_point, 7, (0, 255, 0), -1)
            img = cv2.arrowedLine(img, start_point, end_point, (0, 0, 0), thickness, tipLength=tiplength)
            img = cv2.circle(img, start_point, 7, (0, 0, 255), -1)
            # if std_dev is not None:
            #     img = cv2.circle(img, end, 7, (0, 0, 255), -1)                

        new_lf.img = img
        cv2.imwrite(out_path, new_lf.img)
        return new_lf.img


    def affine_trans(self, kp, data_name=None, matrix=None):
    #    src = np.float32([kp[0], kp[16], kp[27], kp[33], kp[39], kp[42], kp[36], kp[45]])
    #     dst = np.array([[-400,-400], [400, -400], [0, -400], [0, 0],[-100,-300],[100,-300]]) these points shifted by [+700,+1300] in the new dst
    #    dst = np.float32([[300,900], [1100, 900], [700, 915], [700, 1300],[600,920],[800,920], [400, 910], [1000, 910]])
    #     dst = np.float32([[300,900], [1100, 900], [700, 1300]])
    #     tform = AffineTransform()
    #     tform.estimate(src, dst)
    #     matrix = []
    #     temp = str(tform).split("[")
        
    #     for i in range(2,5):matrix.extend(np.array((temp[i].split("]")[0].split(",")), dtype=float))
        
        
        # 6 pt registeration
        src = np.float32([kp[0], kp[16], kp[27], kp[33], kp[39], kp[42]])
        dst = np.float32([[300,900], [1100, 900], [700, 900], [700, 1300],[600,1000],[800,1000]])
        
        # 3 pt registeration
    #     src = np.float32([kp[0], kp[16], kp[33]])
    #     dst = np.float32([[300,1000], [1100, 1000],[700,1300]])
        # 3 pt is not good enough for a face normalization, using more points means more accurate to facial normalization
        
        if data_name=="bp4d":
            src = np.float32([kp[27], kp[33], kp[39], kp[42], kp[36], kp[45]])
            dst = np.float32([[700, 900], [700, 1300],[600,1000],[800,1000], [400, 1000], [1000, 1000]])

        # src_h = np.concatenate((src.T, np.ones(src.shape[0]).reshape(1,src.shape[0])), axis=0)
        # dst_h = np.concatenate((dst.T, np.ones(dst.shape[0]).reshape(1,dst.shape[0])), axis=0)
        # kp_h = np.concatenate((kp.T, np.ones(kp.shape[0]).reshape(1,kp.shape[0])), axis=0)
        
        tform = AffineTransform()
        tform.estimate(src, dst)
        # if matrix is None: matrix = dst_h@np.linalg.pinv(src_h)
        # kp_h = matrix@kp_h
        # return matrix, kp_h[:2, :].T
    
        matrix = []
        temp = str(tform).split("[")
        for i in range(2,5):matrix.extend(np.array((temp[i].split("]")[0].split(",")), dtype=float))
        
        return np.array(matrix), tform(kp)



    def similarity_trans(self, kp, matrix=None): 
        src = kp[[36,39], :]
        dst = np.float32([[400, 1000], [600, 1000]])
        tform = SimilarityTransform()
        tform.estimate(src, dst)
    #     matrix = []
    #     temp = str(tform).split("[")
    #     for i in range(2,5):matrix.extend(np.array((temp[i].split("]")[0].split(",")), dtype=float))
        kp[17:22, :] = tform(kp[17:22, :])
        kp[36:42, :] = tform(kp[36:42, :])   
    # not correct to rotate and scale eyebrows with respect to eyes
        
        src = kp[[42,45], :]
        dst = np.float32([[800, 1000], [1000, 1000]])
        tform = SimilarityTransform()
        tform.estimate(src, dst)
        kp[22:27, :] = tform(kp[22:27, :])
        kp[42:48, :] = tform(kp[42:48, :]) 
    # not correct to rotate and scale eyebrows with respect to eyes   

        src = kp[[0,16], :]
        dst = np.float32([[300, 1000], [1100, 1000]])
        tform = SimilarityTransform()
        tform.estimate(src, dst)
        kp[0:17, :] = tform(kp[0:17, :])
        
        
        delta_x = 700 - kp[27, 0] 
        delta_y = 1000 - kp[27, 1]
        kp[27:36, 0], kp[27:36, 1] =  kp[27:36, 0]+delta_x, kp[27:36, 1]+delta_y
        
        return np.array(matrix), kp



    def affine_vis(self, inp_path, offset_fig_path, apex_fig_path, pt_list=[0,16,27,33,39,42]):
        df = pd.read_csv(inp_path, index_col=0) 

        cols_list_offset = []
        cols_list_apex = []

        for pt in pt_list:
            cols_list_offset.extend(["Offset_x"+str(pt), "Offset_y"+str(pt), "Offset_x"+str(pt)+"_Normalized", 'Offset_y'+str(pt)+"_Normalized"])
            cols_list_apex.extend(["Apex_x"+str(pt), "Apex_y"+str(pt), "Apex_x"+str(pt)+"_Normalized", 'Apex_y'+str(pt)+"_Normalized"])


        for frame_type in ["Offset", "Apex"]:

            if frame_type=="Offset": cols_list = cols_list_offset
            else: cols_list = cols_list_apex

            vis = df[cols_list]

            plt.figure()
            norm_list = ["", "_Normalized"]

            r = np.arange(0,256, 256/len(pt_list),dtype=np.int16)
            g = np.arange(0,256, 256/len(pt_list),dtype=np.int16)
            b = np.arange(0,256, 256/len(pt_list),dtype=np.int16)


            for norm in norm_list:
                plt.subplot(2, 1, norm_list.index(norm)+1)
                color = ['blue','magenta','green','yellow','brown', 'grey']

                handles = []
                for i, j in zip(pt_list, color):
                    handles.append(mpatches.Patch(color=[r[i], g[i], b[i]], label=frame_type+"_x"+str(i)+norm))
                    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
            
                if norm=="_Normalized":
                    x_list = [val for val in cols_list if frame_type+"_x" in val and norm in val]
                    y_list = [val for val in cols_list if frame_type+"_y" in val and norm in val]
                else:
                    x_list = [val for val in cols_list if frame_type+"_x" in val and "_Normalized" not in val]
                    y_list = [val for val in cols_list if frame_type+"_y" in val and "_Normalized" not in val] 

                min_x, min_y, max_x, max_y = int(np.amin(np.array(df[x_list]))), int(np.amin(np.array(df[y_list]))), int(np.amax(np.array(df[x_list]))), int(np.amax(np.array(df[y_list])))
                plt.xlim(int(min_x)-100, int(max_x)+100)
                plt.ylim(int(max_y)+100, int(min_y)-100)
                for pt in pt_list:
                    y_index = int(pt_list.index(pt))
                    x = vis[frame_type+"_x"+str(pt)+norm]
                    y = vis[frame_type+"_y"+str(pt)+norm]
                    plt.scatter(x, y, color=color[y_index])

                # std_points = np.array([[300,900], [1100, 900], [700, 900], [700, 1300],[600,1000],[800,1000]])
                std_points = np.array([[300,900], [1100, 900], [700, 1300]])
                plt.scatter(std_points[:,0], std_points[:, 1],color='black')

            if frame_type=="Offset":
                # plt.show()
                plt.savefig(offset_fig_path, dpi=300, bbox_inches='tight')
            else:
                # plt.show()
                plt.savefig(apex_fig_path, dpi=300, bbox_inches='tight')



