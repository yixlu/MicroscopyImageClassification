import cv2
import numpy as np

class image_preprocessing:
    """
    Preprocessing input microscopic images: split channels, define region of interest (ROI), normalize grayscale values.
    - split_channels: Input RGB images are split into blue, green, and red channels.
    - ROI: Binary masks are constructed using red channel since it gives the contour of a cell,
    which is our region of interest.Foreground:255, background: 0. The binary mask is superimposed on the green channel
    so that pixels in foreground maintains its original value, while those in background are reduced to 0.
    - image_normalize: Normalize the images to the range 0-255 with a specified option.
    """
    def __init__(self,split=True):
        """
        Constructor of image preprocessing object.
        :param split (bool): whether to split the rgb image to three channels. Default to True.
        If False, it will be converted to grayscale.
        """
        self.split=split

    def split_channels(self,src):
        """
        Split the input RGB image into three channels. Return single channel images.
        :param src: a list of multi-channel arrays
        :return: A tuple of two lists: one containing single-channel arrays as the working images
        (green channel or grayscale), the other contains single-channel arrays to be used for mask generation
        (red channel). Blue channel is discarded since its empty.
        """
        b = [None]*len(src)
        g = [None]*len(src)
        r = [None]*len(src)
        for (i,img) in enumerate(src):
            b[i],g[i],r[i]=cv2.split(img)
        self.for_mask = r
        if self.split==True:
            self.img = g
        else:
            gray = [None]*len(src)
            for (i,img) in enumerate(src):
                gray[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.img = gray
        return (self.img, self.for_mask)

    def ROI(self,src=None,for_mask=None,ksize_g=(5, 5), ksize_m=(3, 3)):
        """
        Generate a binary mask from red channel and apply to green channel as region of interest (ROI).
        :param src: (a list of single-channel arrays) Source image to be superimposed with constructed mask.
        :param for_mask: (a list of single-channel arrays) Used to construct binary mask.
        Its length must be equal to that of source image list.
        :param ksize_g: (tuple of ints) kernel size for Gaussian blur. ksize width and height can be different,
        but both have to be positive and odd
        :param ksize_m: (tuple of ints) kernel size for morphological transformation of the binary mask.
        :return:
            - mask: (a list of single-channel arrays) a binary mask where foreground (value 255) defines ROI.
            _ img_masked: (a list of single-channel arrays) masked source image where only pixel values within defined
            ROI are maintained.
        """
        if src:
            self.img = src
        else:
            src=self.img
        if for_mask is None:
            for_mask=self.for_mask
        assert len(src)==len(for_mask), "Number of source images and number of for_mask images do not equal."

        self.mask = [None]*len(for_mask)
        self.img_masked = [None]*len(src)
        for (i,(g,r)) in enumerate(zip(src,for_mask)):
            # construct binary masks
            # Apply Gaussian filter and Otsu thresholding to for_mask images
            blur = cv2.GaussianBlur(r, ksize_g, 0)
            ret, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # remove noise
            kernel = np.ones(ksize_m, np.uint8)
            opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
            # dilate and make final mask
            mask = cv2.dilate(opening, kernel, iterations=3)
            self.mask[i] = mask
            # Filter source images with constructed masks
            img_copy = g.copy()
            img_copy[mask == 0] = 0
            self.img_masked[i] = img_copy
        return (self.mask,self.img_masked)

    def image_normalize(self,option,src=None,mask=None,offset=2.5):
        """
        Perform min-max normalization on input images with a specified option.
        :param option: 'whole','ROI',or 'ROI_on_whole'.
            - 'whole': min-max normalization in range (0,255) is applied to the whole image.
            - 'ROI': min-max normalization is only applied to the ROI.
            - 'ROI_on_whole': Clipping the whole image to the range (mean - offset * STD, mean + offset * STD), where
            mean and STD is calculated for ROI. Then perform min-max normalization.
        :param src: (a list of single-channel arrays) source images.
        If not specified, must call the split_channels or ROI function in class image_preprocessing first.
        :param mask: (a list of single-channel arrays) must be binary images. Number of masks must equal to that of src.
        No need to specify if 'whole' was chosen for option.
        Otherwise, if not specified, you must call the ROI function in class image_preprocessing first.
        :param offset: (int) parameter that only needs to be specified if option 'ROI_on_whole' is chosen.
        :return: (a list of single-channel arrays) normalized images.
        """
        if src is None:
            src = self.img
        self.normalized = [None] * len(src)
        if option == 'whole':
            for (i,img) in enumerate(src):
                self.normalized[i]=cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX)
        elif option == 'ROI':
            if mask is None:
                mask = self.mask
            assert len(src) == len(mask), "Number of source images and number of masks do not equal."
            for (i,(g,m)) in enumerate(zip(src,mask)):
                assert len(np.unique(m)) == 2, "Mask is not binary."
                self.normalized[i]=cv2.normalize(g, 0, 255, norm_type=cv2.NORM_MINMAX, mask=m)
        elif option=='ROI_on_whole':
            if mask is None:
                roi = self.img_masked
            else:
                assert len(src) == len(mask), "Number of source images and number of masks do not equal."
                roi = src.copy()
                for (i,(g,m)) in enumerate(zip(roi,mask)):
                    assert len(np.unique(m)) == 2, "Mask is not binary."
                    g[m == 0] = 0
            for (i,(g,r)) in enumerate(zip(src,roi)):
                # Calculate mean and STD of roi
                mean, STD = cv2.meanStdDev(r)
                # Clip whole image
                clipped = np.clip(g, mean - offset * STD, mean + offset * STD).astype(np.uint8)
                # Normalize to range
                self.normalized[i] = cv2.normalize(clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
        return self.normalized
