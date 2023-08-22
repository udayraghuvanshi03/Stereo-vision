import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

def find_fundamental_matrix(pts_src, pts_dst):
    pts_src=np.array(pts_src)
    pts_dst = np.array(pts_dst)
    ones_arr_src=np.ones((pts_src.shape[0],1))
    ones_arr_dst=np.ones((pts_dst.shape[0],1))
    pts_src=np.hstack((pts_src,ones_arr_src))
    pts_dst=np.hstack((pts_dst,ones_arr_dst))
    mean_src=np.mean(pts_src[:,:2],axis=0)
    S_src=np.sqrt(2)/np.std(pts_src[:,:2])
    T_src=np.array([[S_src, 0, -S_src*mean_src[0]],[0,S_src,-S_src*mean_src[1]],[0,0,1]])
    pts_src=np.dot(T_src,np.transpose(pts_src))
    pts_src=np.transpose(pts_src)
    mean_dst = np.mean(pts_dst[:, :2], axis=0)
    S_dst = np.sqrt(2)/ np.std(pts_dst[:, :2])
    T_dst = np.array([[S_dst, 0, -S_dst * mean_dst[0]], [0, S_dst, -S_dst * mean_dst[1]], [0, 0, 1]])
    pts_dst=np.dot(T_dst,np.transpose(pts_dst))
    pts_dst=np.transpose(pts_dst)
    # forming A matrix
    a_matrix = np.zeros((len(pts_src), 9))
    for i in range(len(pts_src)):
        a_matrix[i, 0] = pts_src[i][1] * pts_dst[i][1]
        a_matrix[i, 1] = pts_src[i][1] * pts_dst[i][0]
        a_matrix[i, 2] = pts_src[i][1]
        a_matrix[i, 3] = pts_src[i][0] * pts_dst[i][1]
        a_matrix[i, 4] = pts_src[i][0] * pts_dst[i][0]
        a_matrix[i, 5] = pts_src[i][0]
        a_matrix[i, 6] = pts_dst[i][1]
        a_matrix[i, 7] = pts_dst[i][0]
        a_matrix[i, 8] = 1
    u, d, u_transpose = np.linalg.svd(a_matrix, full_matrices=True)
    f = np.transpose(u_transpose[-1].reshape(3, 3))
    u_f, d_f, v_f_transpose = np.linalg.svd(f, full_matrices=True)
    d_f[-1] = 0
    f = np.dot(u_f,np.dot(np.diag(d_f), v_f_transpose))
    f=np.dot(np.transpose(T_dst),np.dot(f,T_src))
    # f = f / f[2, 2]
    return f

class StereoVision:
    def __init__(self, array_of_images: list, ransac_iterations, type_of_derivative_filter="sobel",
                 hc_window_size=(5, 5),ncc_threshold=100000, ncc_window=(7, 7)):
        self.array_of_images = array_of_images
        self.type_of_derivative_filter = type_of_derivative_filter
        self.window_size = hc_window_size
        self.threshold = ncc_threshold
        self.ncc_window = ncc_window
        self.iterations = ransac_iterations

    def derivative(self):
        i_x = []
        i_y = []
        for image in self.array_of_images:
            image = cv2.boxFilter(image, -1, (3, 3))
            if self.type_of_derivative_filter.lower() == "sobel":
                sobel_mask_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                sobel_mask_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                i_x.append(cv2.filter2D(image, ddepth=-1, kernel=sobel_mask_x))
                i_y.append(cv2.filter2D(image, ddepth=-1, kernel=sobel_mask_y))
            elif self.type_of_derivative_filter.lower() == "prewitt":
                prewitt_mask_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                prewitt_mask_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                i_x.append(cv2.filter2D(image, ddepth=-1, kernel=prewitt_mask_x))
                i_y.append(cv2.filter2D(image, ddepth=-1, kernel=prewitt_mask_y))
            else:
                print("Incorrect input")
                return
        return i_x, i_y

    def harris_corner_detector(self, k=0.06):
        i_x, i_y = self.derivative()
        harris_r_array = []
        for ind in range(len(i_x)):
            fx = i_x[ind]
            fy = i_y[ind]
            harris_r = np.zeros(np.shape(fx))
            rows, columns = np.shape(fx)[0], np.shape(fx)[1]
            for row in range(rows):
                for column in range(columns):
                    i = row - self.window_size[0] // 2
                    j = column - self.window_size[0] // 2
                    i_x_2 = 0
                    i_y_2 = 0
                    i_x_y = 0
                    while i <= row + self.window_size[0] // 2:
                        while j <= column + self.window_size[0] // 2:
                            if 0 < i < rows and 0 < j < columns:
                                i_x_2 += np.square(fx[i][j])
                                i_y_2 += np.square(fy[i][j])
                                i_x_y += fx[i][j] * fy[i][j]
                            j += 1
                        i += 1
                    m = np.array([[i_x_2, i_x_y], [i_x_y, i_y_2]])
                    r_score = np.linalg.det(m) - k * (np.square(m[0][0] + m[1][1]))
                    if self.threshold < r_score:
                        harris_r[row][column] = r_score
            harris_r_array.append(harris_r)
        return harris_r_array

    def non_max_supression(self):
        harris_r_array = self.harris_corner_detector()
        nms_harris_arr = []
        pix_dist = 5
        h, w = np.shape(harris_r_array[0])
        for ind in harris_r_array:
            nms_harris = np.zeros((h, w))
            for i in range(0, h, 64):
                for j in range(0, w, 64):
                    r_array = []
                    for m in range(i + pix_dist, i + 64 - pix_dist):
                        for n in range(j + pix_dist, j + 64 - pix_dist):
                            if m < h and n < w:
                                if ind[m, n] > 0 and ind[m, n] == np.max(
                                        ind[m - pix_dist:m + pix_dist + 1, n - pix_dist:n + pix_dist + 1]):
                                    r_array.append((ind[m, n], m, n))
                    if len(r_array) < 10:
                        for p in range(len(r_array)):
                            nms_harris[r_array[p][1], r_array[p][2]] = r_array[p][0]
                    else:
                        r_array = sorted(r_array, reverse=True)
                        for p in range(10):
                            nms_harris[r_array[p][1], r_array[p][2]] = r_array[p][0]

            nms_harris_arr.append(nms_harris)
        return nms_harris_arr

    def find_correspondences(self):
        corners_array = self.non_max_supression()
        corners_0 = corners_array[0]
        corners_1 = corners_array[1]
        rows, columns = np.shape(corners_0)[0], np.shape(corners_0)[1]
        correspondences_picture0 = []
        correspondences_picture1 = []
        for row in range(rows):
            for column in range(columns):
                if corners_0[row][column] > 0:
                    a = row - (self.ncc_window[0] // 2)
                    b = row + (self.ncc_window[0] // 2) + 1
                    c = column - (self.ncc_window[0] // 2)
                    d = column + (self.ncc_window[0] // 2) + 1
                    if 0 < a < rows - self.ncc_window[0] and 0 < c < columns - self.ncc_window[0]:
                        template = np.array(self.array_of_images[0][a:b, c:d])
                        max_ncc = 0
                        max_row = None
                        max_column = None
                        for row2 in range(rows):
                            for column2 in range(columns):
                                if corners_1[row2][column2] > 0:
                                    a2 = row2 - (self.ncc_window[0] // 2)
                                    b2 = row2 + (self.ncc_window[0] // 2) + 1
                                    c2 = column2 - (self.ncc_window[0] // 2)
                                    d2 = column2 + (self.ncc_window[0] // 2) + 1
                                    if 0 < a2 < rows - self.ncc_window[0] and 0 < c2 < columns - self.ncc_window[0]:
                                        match_to = np.array(self.array_of_images[1][a2:b2, c2:d2])
                                        f = (match_to - match_to.mean()) / (match_to.std() * np.sqrt(match_to.size))
                                        g = (template - template.mean()) / (template.std() * np.sqrt(template.size))
                                        product = f * g
                                        stds = np.sum(product)
                                        if stds > max_ncc:
                                            max_ncc = stds
                                            max_row = row2
                                            max_column = column2
                        if max_row is not None:
                            if max_ncc > 0.90:
                                correspondences_picture0.append([row, column])
                                correspondences_picture1.append([max_row, max_column])
        return correspondences_picture0, correspondences_picture1

    def feature_matching(self, img_l, img_r, in_0, in_1):
        pts_0, pts_1 = in_0,in_1
        img_l = np.copy(img_l)
        img_r = np.copy(img_r)
        shift = np.shape(img_r)[1]
        full_img = np.concatenate((img_l, img_r), axis=1)

        for i in range(len(pts_0)):
            color = list(np.random.random_sample(size=3) * 256)
            cv2.line(full_img, (pts_0[i][1], pts_0[i][0]), (pts_1[i][1] + shift, pts_1[i][0]),
                     color, thickness=1)

        return full_img

    def ransac(self):
        corres_0, corres_1 = self.find_correspondences()
        max_inliers = 0
        fundamental_matrix_with_all = None
        max_inliers_0 = []
        max_inliers_1 = []
        for it in range(self.iterations):
            random_sampled_indx = []
            pts_src = []
            pts_dst = []
            i = 0
            while i < 8:
                random_sample = int(np.random.randint(0, high=len(corres_0) - 1, size=1, dtype=int))
                if random_sample not in random_sampled_indx:
                    pts_src.append(corres_0[random_sample])
                    pts_dst.append(corres_1[random_sample])
                    random_sampled_indx.append(random_sample)
                    i += 1
            fundamental_matrix = find_fundamental_matrix(pts_src, pts_dst)
            j = 0
            inliers = 0
            inliers_0 = []
            inliers_1 = []
            while j < len(corres_0):
                pl = np.transpose(np.array([[corres_0[j][1], corres_0[j][0], 1]]))
                pr_t = np.array([[corres_1[j][1], corres_1[j][0], 1]])
                est = np.dot(pr_t,np.dot(fundamental_matrix,pl))
                # print(est)
                if abs(est) < 0.007:
                    inliers += 1
                    inliers_0.append(corres_0[j])
                    inliers_1.append(corres_1[j])
                j += 1
            if inliers > max_inliers:
                max_inliers = inliers
                max_inliers_0 = inliers_0
                max_inliers_1 = inliers_1
                if max_inliers > 7:
                    fundamental_matrix_with_all = find_fundamental_matrix(max_inliers_0, max_inliers_1)
        return fundamental_matrix_with_all, max_inliers_0, max_inliers_1

    def disp_img(self, img1, img2):
        nms_harris = self.non_max_supression()
        rgb_imgs_arr = [img1, img2]
        corner_imgs = []
        for k in range(len(nms_harris)):
            rgb = np.copy(rgb_imgs_arr[k])
            r, c = np.shape(rgb)[0], np.shape(rgb)[1]
            for i in range(r):
                for j in range(c):
                    if nms_harris[k][i, j] > 0:
                        rgb[i, j] = [0, 0, 255]
            corner_imgs.append(rgb)
        return corner_imgs

def disparity_map(array_of_images, fund_matrix):
    left_img=array_of_images[0]
    right_img=array_of_images[1]
    rows,cols=left_img.shape[0],left_img.shape[1]
    window_size=7
    block=window_size//2
    hori_disp_img=np.zeros_like(left_img,np.uint8)
    vert_disp_img=np.zeros_like(left_img,np.uint8)
    disp_color=np.zeros_like(left_img, np.uint8)
    epi_lines_right=np.zeros((rows,cols,3))
    max_disp=50
    for i in range(rows):
        for j in range(cols):
            pt_l=np.transpose(np.array([[j,i,1]]))
            l_r=np.dot(np.transpose(fund_matrix),pt_l)
            epi_lines_right[i, j, :] =l_r.reshape(-1)
    for row in range(block,rows-block):
        for col in range(block,cols-block):
            a, b, c = epi_lines_right[row,col]
            left_win=left_img[row-block:row+block+1,col-block:col+block+1]
            max_similarity = float('inf')
            best_match_col=-1
            best_match_row=-1
            for k in range(max_disp):
                y_r = int((-a * (col+k) - c) / b)
                right_win = right_img[row - block:row+ block + 1, col - block - k:col + block + 1 - k]
                # right_win=right_img[y_r-block:y_r+block+1,col-block+k:col+block+1+k]
                if left_win.shape == right_win.shape:
                    sad = np.sum(abs(left_win - right_win))
                    if sad < max_similarity:
                        max_similarity = sad
                        best_match_col = k
                        best_match_row = row
            hori_disp=col-(col-best_match_col)
            vert_disp=row-(row-best_match_row)
            hori_disp = np.clip(hori_disp, 0, 255)
            vert_disp = np.clip(vert_disp, 0, 255)
            hori_disp_img[row, col] =hori_disp
            vert_disp_img[row,col]=vert_disp
            disp_color[row,col]=np.sqrt(np.sum((hori_disp**2,vert_disp**2)))
    disp_color=cv2.normalize(hori_disp_img, None, 0, 1, cv2.NORM_MINMAX)
    hue = disp_color*0.5
    saturation = np.ones_like(disp_color,dtype=np.float32)
    disparity_hsv = np.stack((hue, saturation, saturation), axis=-1)
    disparity_hsv=(255*disparity_hsv).astype(np.uint8)
    disparity_hsv = cv2.cvtColor(disparity_hsv, cv2.COLOR_HSV2BGR)
    hori_disp_img=cv2.normalize(hori_disp_img,None, 0, 255, cv2.NORM_MINMAX)
    vert_disp_img= cv2.normalize(vert_disp_img, None, 0, 255, cv2.NORM_MINMAX)
    return hori_disp_img,vert_disp_img,disparity_hsv

def drawlines(left_img,right_img,pts1, pts2,fund_matrix):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        img1 = right_img
        img2 = left_img
        r, c = img1.shape[0],img1.shape[1]
        for i in range(len(pts1)):
            pt_l = np.transpose(np.array([[pts1[i][1], pts1[i][0], 1]]))
            l_r = np.dot(np.transpose(fund_matrix), pt_l)
            color = list(np.random.random_sample(size=3) * 256)
            x0, y0 = map(int, [0, -l_r[2] / l_r[1]])
            x1, y1 = map(int, [c, -(l_r[2] + l_r[0] * c) / l_r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        for j in range(len(pts2)):
            pr_t=np.array([pts2[j][1],pts2[j][0],1])
            l_l=np.dot(pr_t,fund_matrix)
            color = list(np.random.random_sample(size=3) * 256)
            x0_l, y0_l = map(int, [0, -l_l[2] / l_l[1]])
            x1_l, y1_l = map(int, [c, -(l_l[2] + l_l[0] * c) / l_l[1]])
            img2 = cv2.line(img2, (x0_l, y0_l), (x1_l, y1_l), color, 1)
        return img1,img2


if __name__=='__main__':
    data1_l = r"C:\Users\udayr\PycharmProjects\CVfiles\project3\images\cast-left-1.jpg"
    data1_r = r"C:\Users\udayr\PycharmProjects\CVfiles\project3\images\cast-right-1.jpg"
    data1 = [data1_l, data1_r]

    data2_l = r"C:\Users\udayr\PycharmProjects\CVfiles\project3\images\image-3.jpeg"
    data2_r = r"C:\Users\udayr\PycharmProjects\CVfiles\project3\images\image-4.jpeg"
    data2 = [data2_l, data2_r]
    imgs_arr_gray_data2 = []
    imgs_arr_rgb_data2 = []
    for img_path in data2:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read and convert the image to grayscale
        imgs_arr_gray_data2.append(np.asarray(img_gray).astype(float))  # Create an array of all the images

        img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read the image as RGB
        imgs_arr_rgb_data2.append(np.asarray(img_rgb))
    stereo_v_data2 = StereoVision(imgs_arr_gray_data2, 10000)
    # bef_rans_0, bef_rans_1=stereo_v_data2.find_correspondences()
    # img_bef=stereo_v_data2.feature_matching(imgs_arr_rgb_data2[0],imgs_arr_rgb_data2[1],bef_rans_0,bef_rans_1)
    # cv2.imshow('img_before',img_bef)
    # cv2.waitKey(0)

    if os.path.exists('fundamental_matrix_data2.npy'):
        F_data2 = np.load('fundamental_matrix_data2.npy')
        in1_data2=np.load('in1_data2.npy')
        in2_data2=np.load('in2_data2.npy')
    else:
        # Compute the fundamental matrix using RANSAC
        F_data2, in1_data2, in2_data2=stereo_v_data2.ransac()
        np.save('fundamental_matrix_data2.npy', F_data2)
        np.save('in1_data2.npy',in1_data2)
        np.save('in2_data2.npy',in2_data2)

    # epi_lines_r,epi_lines_l = drawlines(imgs_arr_rgb_data2[0],imgs_arr_rgb_data2[1],in1_data2, in2_data2, F_data2)
    # full_img=np.concatenate((epi_lines_l,epi_lines_r),axis=1)
    # cv2.imshow('Epilines on left and right image',full_img)
    # cv2.waitKey(0)

    # retBool, rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(in1_data2, in2_data2, F_data2, (450, 375))
    # left_img_rect = cv2.warpPerspective(imgs_arr_gray_data2[0], rectmat1, (450, 375))
    # right_img_rect=cv2.warpPerspective(imgs_arr_gray_data2[1], rectmat2, (450, 375))
    # cv2.imshow('img',left_img_rect.astype(np.uint8))
    # cv2.waitKeyEx(0)

    # print(left_img_rect)
    # cv2.imshow('img',full_img)
    # cv2.waitKey(0)
    # cv2.imshow('img', right_img_rect)
    # cv2.waitKey(0)

    hori_disp_img,vertical_disp_img,colored_disp=disparity_map(imgs_arr_gray_data2, F_data2)
    full_img = np.concatenate((hori_disp_img, vertical_disp_img), axis=1)
    cv2.imshow('img',hori_disp_img)
    cv2.waitKey(0)
    cv2.imshow('img',vertical_disp_img)
    cv2.waitKey(0)
    cv2.imshow('img',colored_disp)
    cv2.waitKey(0)
    # img_aft=stereo_v_data2.feature_matching(imgs_arr_rgb_data2[0],imgs_arr_rgb_data2[1],in1_data2,in2_data2)
    # cv2.imshow('img_before',img_aft)
    # cv2.waitKey(0)
    #
    imgs_arr_gray_data1 = []
    imgs_arr_rgb_data1 = []
    for img_path in data1:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read and convert the image to grayscale
        imgs_arr_gray_data1.append(np.asarray(img_gray).astype(float))  # Create an array of all the images

        img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read the image as RGB
        imgs_arr_rgb_data1.append(np.asarray(img_rgb))
    stereo_v_data1 = StereoVision(imgs_arr_gray_data1, 10000)
    # F,in1,in2=stereo_v_data1.ransac()
    # img_bef=stereo_v_data1.feature_matching(imgs_arr_rgb_data1[0],imgs_arr_rgb_data1[1],bef_rans_0,bef_rans_1)
    # cv2.imshow('img_before',img_bef)
    # # cv2.waitKey(0)

    if os.path.exists('fundamental_matrix_data1.npy'):
        F_data1 = np.load('fundamental_matrix_data1.npy')
        in1_data1 = np.load('in1_data1.npy')
        in2_data1 = np.load('in2_data1.npy')
    else:
        # Compute the fundamental matrix using RANSAC
        F_data1, in1_data1, in2_data1 = stereo_v_data1.ransac()
        np.save('fundamental_matrix_data1.npy', F_data1)
        np.save('in1_data1.npy', in1_data1)
        np.save('in2_data1.npy', in2_data1)

    # img_aft=stereo_v_data1.feature_matching(imgs_arr_rgb_data1[0],imgs_arr_rgb_data1[1],in1_data1,in2_data1)
    # cv2.imshow('img_before',img_aft)
    # cv2.waitKey(0)
    hori_disp_img,vertical_disp_img,colored_disp=disparity_map(imgs_arr_gray_data1, F_data1)
    full_img = np.concatenate((hori_disp_img, vertical_disp_img), axis=1)
    cv2.imshow('img', hori_disp_img)
    cv2.waitKey(0)
    cv2.imshow('img', vertical_disp_img)
    cv2.waitKey(0)
    cv2.imshow('img', colored_disp)
    cv2.waitKey(0)
    # epi_lines_r,epi_lines_l = drawlines(imgs_arr_gray_data1[0],imgs_arr_gray_data1[1],in1_data1, in2_data1, F_data1)
    # full_img=np.concatenate((epi_lines_l.astype(np.uint8),epi_lines_r.astype(np.uint8)),axis=1)
    # cv2.imshow('img',full_img)
    # cv2.waitKey(0)
