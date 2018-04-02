# Single-Image-Super-Resolution
A list of resources for example-based single image super-resolution, inspired by [Awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision) and [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision) .

By Yapeng Tian and Yunlun Zhang (if you have any suggestions, please contact us! Email: yapengtian@rochester.edu OR yulun100@gmail.com).

## Example-based methods

### Early learning-based methods

[1] Freeman, William T and Pasztor, Egon C and Carmichael, Owen T, Learning low-level vision, IJCV, 2000. [[Paper]](http://people.csail.mit.edu/billf/papers/TR2000-05.pdf) ([Freeman](billf.mit.edu) et al. first presented example-based or learning-based super-resolution framework - learn relationships between low-resolution image patches and its high-resolution counterparts.)

[2] Freeman, William T and Jones, Thouis R and Pasztor, Egon C, Example-based super-resolution, IEEE Computer graphics and Applications, 2002.    [[Paper]](http://www.merl.com/publications/docs/TR2001-30.pdf) 

[3] Chang, Hong and Yeung, Dit-Yan and Xiong, Yimin, Super-resolution through neighbor embedding, CVPR, 2004. [[Paper]](http://repository.ust.hk/ir/bitstream/1783.1-2284/1/yeung.cvpr2004.pdf) [[Code]](http://www.jdl.ac.cn/user/hchang/publication.htm) (The idea that low-resolution patches and corresponding high-resolution patches share similar local geometries highly influences the subsequent coding-based or dictionary-based methods.)

### Sparsity-based methods
[1] Yang, Jianchao and Wright, John and Huang, Thomas S and Ma, Yi, Image super-resolution via sparse representation, IEEE trans. image processing 2010. [[paper]](http://ieeexplore.ieee.org/document/5466111/?arnumber=5466111) [[Code]](http://www.ifp.illinois.edu/~jyang29/) (SCSR: Classical sparsity-based SISR method - use sparse coding technique to learn low-resolution and high-resolution dictionaries.)

[2] Zeyde, Roman and Elad, Michael and Protter, Matan, On single image scale-up using sparse-representations, International conference on curves and surfaces, 2010. [[Paper]](http://www.cs.technion.ac.il/~elad/publications/conferences/2010/ImageScaleUp_LNCS.pdf) [[Code]](http://www.cs.technion.ac.il/~elad/software/)  (Low dimension feature speeds up the algorithm. Many sparsity-based image restoration techniques can be found in Prof. [Elad](http://www.cs.technion.ac.il/~elad/index.html)'s Website!) 

[3] Weisheng Dong, Lei Zhang, Guangming Shi, and Xiaolin Wu, Image Deblurring and Super-resolution by Adaptive Sparse Domain Selection and Adaptive Regularization, TIP, 2011. [[Website]](http://www4.comp.polyu.edu.hk/~cslzhang/ASDS_AReg.htm) (Clustering is a very effective trick and local and nonlocal regularization terms are very powerful! Other good sparsity-based super-resolution methods can be found in Prof. [Lei Zhang](http://www4.comp.polyu.edu.hk/~cslzhang/)'s and [Weisheng Dong](http://see.xidian.edu.cn/faculty/wsdong/)'s Website!)

[4] Peleg, Tomer and Elad, Michael, A statistical prediction model based on sparse representations for single image super-resolution, TIP, 2014. [[Paper]](http://www.cs.technion.ac.il/~elad/publications/journals/2013/SingleImageSR_TIP.pdf) [[Code]](http://www.cs.technion.ac.il/~elad/software/) (Predict the relationships between Low-resolution and high-resolution representation coefficients.)

### Super-resolution via self-examplars

[1] Daniel Glasner, Shai Bagon and Michal, Irani, Super-Resolution from a Single Image, ICCV, 2009. [[Paper]](http://www.wisdom.weizmann.ac.il/~vision/single_image_SR/files/single_image_SR.pdf)

[2] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja, "Single Image Super-Resolution from Transformed Self-Exemplars", CVPR, 2015. [[Project]](https://github.com/jbhuang0604/SelfExSR).

### Locally Linear Regression

[1] Gu, Shuhang and Sang, Nong and Ma, Fan, Fast Image Super Resolution via Local Regression, ICPR, 2012. [[Paper]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6460827) (Kmeans clusetering + ridge regression)

[2] Timofte, Radu and De Smet, Vincent and Van Gool, Luc, Anchored neighborhood regression for fast example-based super-resolution, ICCV, 2013. [[Paper]](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Timofte_Anchored_Neighborhood_Regression_2013_ICCV_paper.pdf) [[Website]](http://www.vision.ee.ethz.ch/~timofter/) (ANR method)

[3] Yang, Chih-Yuan and Yang, Ming-Hsuan, Fast direct super-resolution by simple functions, ICCV, 2013. [[Paper]](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Yang_Fast_Direct_Super-Resolution_2013_ICCV_paper.pdf) [[Website]](https://eng.ucmerced.edu/people/cyang35/ICCV13/ICCV13.html) 

[4] Timofte, Radu and De Smet, Vincent and Van Gool, Luc, A+: Adjusted anchored neighborhood regression for fast super-resolution, ACCV, 2014. [[Paper]](https://pdfs.semanticscholar.org/ca57/66b91da4903ad6f6d40a5b31a3ead1f7f6de.pdf) [[Website]](http://www.vision.ee.ethz.ch/~timofter/) (More data and better performance!)

[5] Schulter, Samuel and Leistner, Christian and Bischof, Horst, Fast and accurate image upscaling with super-resolution forests, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schulter_Fast_and_Accurate_2015_CVPR_paper.pdf) [[Code]](http://lrs.icg.tugraz.at/members/schulter#software)

[6] Salvador, Jordi, and Eduardo Pérez-Pellitero, Naive Bayes Super-Resolution Forest, ICCV, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Salvador_Naive_Bayes_Super-Resolution_ICCV_2015_paper.pdf) [[Website]](http://perezpellitero.github.io/) (Very fast!)

[7] E. Pérez-Pellitero and J. Salvador and J. Ruiz-Hidalgo and B. Rosenhahn, PSyCo: Manifold Span Reduction for Super Resolution, CVPR, 2016. [[Paper]](http://perezpellitero.github.io/documents/PerezPellitero2016Cvpr.pdf) [[Website]](http://perezpellitero.github.io/) (choose a better distance metric!)

[8] Timofte, Radu and Rothe, Rasmus and Van Gool, Luc, Seven Ways to Improve Example-Based Single Image Super Resolution, CVPR, 2016. [[Website]](http://www.vision.ee.ethz.ch/~timofter/)

### Deep Architectures

[1] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Learning a deep convolutional network for image super-resolution, ECCV, 2014. [[Website]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) (first introduce CNN to solve single image super-resolution.)

[2] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Image Super-Resolution Using Deep Convolutional Networks, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2016. [[Website]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) (use more training data and achieve better SR performance.) [[Keras]](http://github.com/YapengTian/SRCNN-Keras)

[3] Wang, Zhaowen and Liu, Ding and Yang, Jianchao and Han, Wei and Huang, Thomas, Deep networks for image super-resolution with sparse prior, ICCV, 2015. [[Website]](http://www.ifp.illinois.edu/~dingliu2/iccv15/) 

[4] Jimmy SJ. Ren, Li Xu, Qiong Yan, Wenxiu Sun, Shepard Convolutional Neural Networks, NIPS, 2015. [[Paper]](https://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf) [[Code]](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/Shepard_CNN)

[5] Shi, Wenzhe and Caballero, Jose and Huszar, Ferenc and Totz, Johannes and Aitken, Andrew P. and Bishop, Rob and Rueckert, Daniel and Wang, Zehan, Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, CVPR, 2016. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)

[6] Kim, Jiwon and Kwon Lee, Jung and Mu Lee, Kyoung, Accurate Image Super-Resolution Using Very Deep Convolutional Networks, CVPR, 2016. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf) [[Code]](http://github.com/huangzehao/caffe-vdsr) [[website]](http://cv.snu.ac.kr/?page_id=60)

[7] Kim, Jiwon and Kwon Lee, Jung and Mu Lee, Kyoung, Deeply-Recursive Convolutional Network for Image Super-Resolution, CVPR, 2016. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.pdf) [[website]](http://cv.snu.ac.kr/?page_id=60)

[8] Chao Dong, Chen Change Loy, Xiaoou Tang, Accelerating the Super-Resolution Convolutional Neural Network, ECCV, 2016. [[Paper]](https://arxiv.org/pdf/1608.00367.pdf) [[Code]](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)

[9] Justin Johnson, Alexandre Alahi, Fei-Fei Li, Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV, 2016, [[Website]](http://cs.stanford.edu/people/jcjohns/) (Perceptual Loss)

[10] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, arXiv, 2016. [[Paper]](https://arxiv.org/pdf/1609.04802.pdf) (Perceptual Loss, Great Performance!)

[11] Julien Maira, End-to-End Kernel Learning with Supervised Convolutional Kernel Networks, NIPS, 2016. [[Paper]](https://arxiv.org/pdf/1605.06265v1.pdf)

[12] Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang, Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections, arXiv, 2016. [[Paper]](https://arxiv.org/pdf/1606.08921.pdf)

[13] Joan Bruna, Pablo Sprechmann, Yann LeCun, SUPER-RESOLUTION WITH DEEP CONVOLUTIONAL SUFFICIENT STATISTICS, ICLR, 2016. [[Paper]](https://arxiv.org/pdf/1511.05666.pdf) (Perceptual Loss)

[14] Mehdi S. M. Sajjadi, Bernhard Schölkopf, Michael Hirsch, EnhanceNet: Single Image Super-Resolution through Automated Texture Synthesis, ICCV, 2017. [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Sajjadi_EnhanceNet_Single_Image_ICCV_2017_paper.pdf) (adversarial training + Texture matching loss to reduce unnatural textures produced by perceptual loss)

[15] Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, Ferenc Huszár, Amortised MAP Inference for Image Super-resolution, ICLR, 2017. [[Paper]](https://arxiv.org/pdf/1610.04490v3.pdf) (calculate the MAP estimate directly using a convolutional neural network)

[16] Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang, Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution, CVPR, 2017. [[Website]](http://graduatestudents.ucmerced.edu/wlai24/)

[17] K. Zhang, W. Zuo, S. Gu and L. Zhang, "Learning Deep CNN Denoiser Prior for Image Restoration," CVPR, 2017. [[Code]](http://github.com/cszn/ircnn)

[18] Ying Tai, Jian Yang, and Xiaoming Liu. Image Super-Resolution via Deep Recursive Residual Network, CVPR, 2017. [[Code]](https://github.com/tyshiwo/DRRN_CVPR17)

[19] E. Agustsson, R. Timofte, L. Van Gool. Anchored Regression Networks applied to Age Estimation and Super Resolution, ICCV, 2017. [[paper]](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-ICCV-2017.pdf)

[20] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee. Enhanced Deep Residual Networks for Single Image Super-Resolution. CVPRW, 2017. [[paper]](https://arxiv.org/abs/1707.02921)(state-of-the-art) [[Code]](https://github.com/LimBee/NTIRE2017)

[21] Ying Tai, Jian Yang, Xiaoming Liu and Chunyan Xu. MemNet: A Persistent Memory Network for Image Restoration, ICCV, 2017. [[code]](https://github.com/tyshiwo/MemNet)

[22] Radu Timofte et al. NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results, CVPRW, 2017. [[Paper]](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPRW-2017.pdf)

[23] Jin Yamanaka, Shigesumi Kuwashima and Takio Kurita: Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network, ICONIPS, 2017. [[Paper]](https://arxiv.org/pdf/1707.05425.pdf)[[Code]](https://github.com/jiny2001/dcscn-super-resolution) 

[24] Tong Tong, Gen Li, Xiejie Liu, Qinquan Gao. Image Super-Resolution Using Dense Skip Connections. ICCV, 2017. [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

[25] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, Yun Fu. Residual Dense Network for Image Super-Resolution. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1802.08797.pdf) [[code]](https://github.com/yulunzhang/RDN)

[26] Muhammad Haris, Greg Shakhnarovich, and Norimichi Ukita. Deep Back-Projection Networks For Super-Resolution. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1803.02735.pdf) [[code-caffe]](https://github.com/alterzero/DBPN-caffe) [[code-pytorch]](https://github.com/alterzero/DBPN-Pytorch)

[27] Kai Zhang, Wangmeng Zuo, and Lei Zhang. Learning a Single Convolutional Super-Resolution Network for Multiple Degradations. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1712.06116.pdf) [[code]](https://github.com/cszn/SRMD)


