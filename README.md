# Single-Image-Super-Resolution
A list of resources for example-based single image super-resolution, inspired by [Awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision) and [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision) .

By Yapeng Tian and Yunlun Zhang (if you have any suggestions, please contact us! Email: yapengtian@rochester.edu OR yulun100@gmail.com).

Tip: For SR beginners, I recommend you to read some early learning based SISR works which will help understand the problem.

## Survey Paper

[1] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, Jing-Hao Xue. Deep Learning for Single Image Super-Resolution:
A Brief Review. TMM, 2019. [[Paper]](https://arxiv.org/pdf/1808.03344.pdf)

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

[28] Adrian Bulat and Georgios Tzimiropoulos. Super-FAN: Integrated facial landmark localization and super-resolution of real-world low resolution faces in arbitrary poses with GANs. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1712.02765.pdf)

[29] Bjoern Haefner, Yvain Queau, Thomas Möllenhoff, and Daniel Cremers. Fight ill-posedness with ill-posedness: Single-shot variational depth super-resolution from shading. CVPR 2018.

[30] Xintao Wang, Ke Yu, Chao Dong, and Chen-Change Loy. Recovering Realistic Texture in Image Super-resolution by Spatial Feature Modulation. CVPR 2018. [[Paper]](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)

[31] Zheng Hui, Xiumei Wang, and Xinbo Gao. Fast and Accurate Single Image Super-Resolution via Information Distillation Network. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1803.09454.pdf)

[32] Xin Yu, Basura Fernando, Richard Hartley, and Fatih Porikli. Super-Resolving Very Low-Resolution Face Images with Supplementary Attributes. CVPR 2018. [[Paper]](https://basurafernando.github.io/papers/XinYuCVPR18.pdf)

[33] Wei Han, Shiyu Chang, Ding Liu, Michael Witbrock, and Thomas Huang. Image Super-resolution via Dual-state Recurrent Neural Networks. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1805.02704.pdf)

[34] Yu Chen, Ying Tai, Xiaoming Liu, Chunhua Shen, and Jian Yang. FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1711.10703.pdf)

[35] Ying Qu, Hairong Qi, and Chiman Kwan. Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution. CVPR 2018.

[36] Assaf Shocher, Nadav Cohen, and Michal Irani. “Zero-Shot” Super-Resolution using Deep Internal Learning. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1712.06087.pdf)

[37] Younghyun Jo, Seoung Wug Oh, JaeYeon Kang, and Seon Joo Kim. Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation. CVPR 2018.

[38] Weimin Tan, Bo Yan, and Bahetiyaer Bare. Feature Super-Resolution: Make Machine See More Clearly. CVPR 2018.

[39] Mehdi S. M. Sajjadi, Raviteja Vemulapalli, and Matthew Brown. Frame-Recurrent Video Super-Resolution. CVPR 2018. [[Paper]](https://arxiv.org/pdf/1801.04590.pdf)

[40] Yifan Wang, Federico Perazzi, Brian McWilliams, Alexander Sorkine-Hornung, Olga Sorkine-Hornung, Christopher Schroers.
A Fully Progressive Approach to Single-Image Super-Resolution. arXiv, 2018. [[Paper]](https://arxiv.org/pdf/1804.02900.pdf)

[41] Roey Mechrez, Itamar Talmi, Firas Shama, Lihi Zelnik-Manor. Learning to Maintain Natural Image Statistics. arXiv, 2018. [[Paper]](https://arxiv.org/abs/1803.04626) [[Project]](http://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/Contextual/) [[ECCV SR Challenge]](https://www.pirm2018.org/).

[42] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, Yun Fu. Image Super-Resolution Using Very Deep Residual Channel Attention Networks. ECCV 2018. [[Paper]](https://arxiv.org/pdf/1807.02758.pdf) [[code]](https://github.com/yulunzhang/RCAN)

[43] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, Jing-Hao Xue. Deep Learning for Single Image Super-Resolution:
A Brief Review. arxiv, 2018. [[Paper]](https://arxiv.org/pdf/1808.03344.pdf) (a survey paper)

[44] Adrian Bulat, Jing Yang, Georgios Tzimiropoulos. To learn image super-resolution, use a GAN to
learn how to do image degradation first. ECCV, 2018. [[Paper]](https://arxiv.org/pdf/1807.11458.pdf) 

[45] Namhyuk Ahn, Byungkon Kang, Kyung-Ah Sohn. Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network. ECCV 2018. [[Paper]](https://arxiv.org/abs/1803.08664)

[46] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang. Enhanced Super-Resolution Generative Adversarial Networks. ECCV2018 PIRM Workshop. [[Code]](https://github.com/xinntao/ESRGAN)

[47] Seong-Jin Park, Hyeongseok Son, Sunghyun Cho, Ki-Sang Hong. SRFeat: Single Image Super-Resolution with Feature Discrimination. ECCV 2018. [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf)

[48] Subeesh Vasu, Nimisha T. M., A. N. Rajagopalan. Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network. ECCV2018 PIRM Workshop. [[Code]](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)

[49] Vu, Thang and Van Nguyen, Cao and Pham, Trung X. and Luu, Tung M. and Yoo, Chang Dong. Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks. ECCV2018 PIRM Mobile Workshop [[Paper]](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Fast_and_Efficient_Image_Quality_Enhancement_via_Desubpixel_Convolutional_Neural_ECCVW_2018_paper.pdf) [[Code]](https://github.com/thangvubk/FEQE)

[50] Yulun Zhang, Kunpeng Li, Kai Li, Bineng Zhong, Yun Fu. Image Super-Resolution Using Very Deep Residual Channel Attention Networks. ICLR 2019. [[Paper]](https://openreview.net/pdf?id=HkeGhoA5FX) [[code]](https://github.com/yulunzhang/RNAN)

[51] Xintao Wang, Ke Yu, Chao Dong, Xiaoou Tang, Chen Change Loy. Deep Network Interpolation for Continuous Imagery Effect Transition. CVPR 2019. [[Website]](https://xinntao.github.io/projects/DNI)

[52] Xuecai Hu, Haoyuan Mu, Xiangyu Zhang, Zilei Wang, Jian Sun, Tieniu Tan. Meta-SR: A Magnification-Arbitrary Network for Super-Resolution. ArXiv, 2019. [[Paper]](https://arxiv.org/pdf/1903.00875.pdf)

[53] Zhang, Kai and Zuo, Wangmeng and Zhang, Lei. Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels, CVPR 2019. [[Project]](https://github.com/cszn/DPSR)

[54] Xuaner Zhang, Qifeng Chen, Ren Ng, and Vladlen Koltun. Zoom to Learn, Learn to Zoom, CVPR 2019. [[Paper]](http://vladlen.info/papers/zoom.pdf)

[55] Zhen Li, Jinglei Yang, Zheng Liu, Xiaomin Yang, Gwanggil Jeon, and Wei Wu. Feedback Network for Image Super-Resolution, CVPR 2019. [[Paper]](https://arxiv.org/abs/1903.09814)[[Code]](https://github.com/Paper99/SRFBN_CVPR19)

[56] Chang Chen, Zhiwei Xiong, Xinmei Tian, Zheng-Jun Zha, and Feng Wu. Camera Lens Super-Resolution, CVPR 2019. [[Paper]](https://arxiv.org/pdf/1904.03378.pdf)[[Code]](https://github.com/ngchc/CameraSR)

[57] Zhifei Zhang, Zhaowen Wang, Zhe Lin, Hairong Qi. Image Super-Resolution by Neural Texture Transfer. CVPR 2019. [[Paper]](https://arxiv.org/pdf/1903.00834.pdf)

[58] Tao Dai, Jianrui Cai, Yongbing Zhang, Shu-Tao Xia, Lei Zhang. Second-order Attention Network for Single Image Super-resolution. CVPR 2019. [[Project]](https://github.com/daitao/SAN)

[59] Gu, Jinjin and Lu, Hannan and Zuo, Wangmeng and Dong, Chao. Blind Super-Resolution With Iterative Kernel Correction, CVPR 2019. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.pdf)

[60] Xu, Xiangyu and Ma, Yongrui and Sun, Wenxiu. Towards Real Scene Super-Resolution With Raw Images, CVPR 2019. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.pdf)

[61] He, Xiangyu and Mo, Zitao and Wang, Peisong and Liu, Yang and Yang, Mingyuan and Cheng, Jian. ODE-Inspired Network Design for Single Image Super-Resolution. CVPR 2019. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/He_ODE-Inspired_Network_Design_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)

[62] Wei Wang, Ruiming Guo, Yapeng Tian, Wenming Yang. CFSNet: Toward a Controllable Feature Space for Image Restoration, ICCV 2019. [[Paper]](https://arxiv.org/abs/1904.00634)

[63] Jun-Ho Choi, Huan Zhang, Jun-Hyuk Kim, Cho-Jui Hsieh, Jong-Seok Lee. Evaluating Robustness of Deep Image Super-Resolution Against Adversarial Attacks, ICCV, 2019. [[Paper]](https://arxiv.org/pdf/1904.06097.pdf)

[64] Ruofan Zhou, Sabine Süsstrunk. 	Kernel Modeling Super-Resolution on Real Low-Resolution Images. ICCV, 2019. [[Project]](https://ivrlwww.epfl.ch/ruofan/project_KMSR/KMSR.html)

[65] Mohammad Saeed Rad, Behzad Bozorgtabar, Urs-Viktor Marti, Max Basler, Hazim Kemal Ekenel, Jean-Philippe Thiran. SROBB: Targeted Perceptual Loss for Single Image Super-Resolution. ICCV, 2019. [[Paper]](https://arxiv.org/abs/1908.07222)

[66] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, Lei Zhang. Toward Real-World Single Image Super-Resolution: A New Benchmark and a New Model. ICCV, 2019. [[Paper]](https://arxiv.org/pdf/1904.00523.pdf)

[67] Wenlong Zhang, Yihao Liu, Chao Dong, Yu Qiao. 	RankSRGAN: Generative Adversarial Networks With Ranker for Image Super-Resolution. ICCV, 2019.[[Paper]](https://arxiv.org/abs/1908.06382)

[68] Tamar Rott Shaham, Tali Dekel, Tomer Michaeli. SinGAN: Learning a Generative Model from a Single Natural Image. ICCV, 2019. [[Project]](https://github.com/tamarott/SinGAN)

[69] Kai Zhang, Luc Van Gool, Radu Timofte. Deep Unfolding Network for Image Super-Resolution. CVPR 2020. [[Project]](https://github.com/cszn/USRNet)

[70] Cheng Ma, Yongming Rao, Yean Cheng, Ce Chen, Jiwen Lu, Jie Zhou. Structure-Preserving Super Resolution with Gradient Guidance. CVPR 2020. [[Project]](https://arxiv.org/pdf/2003.13081.pdf)

[71] Shunta Maeda. Unpaired Image Super-Resolution using Pseudo-Supervision. CVPR, 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf)

[72] Givi Meishvili, Simon Jenni, Paolo Favaro. Learning to Have an Ear for Face Super-Resolution. CVPR, 2020. [[Project]](https://gmeishvili.github.io/ear_for_face_super_resolution/index.html)

[73] Shady Abu Hussein, Tom Tirer, Raja Giryes. Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers. CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Abu_Hussein_Correction_Filter_for_Single_Image_Super-Resolution_Robustifying_Off-the-Shelf_Deep_Super-Resolvers_CVPR_2020_paper.pdf)

[74] Jie Liu, Wenjie Zhang, Yuting Tang, Jie Tang, Gangshan Wu. Residual Feature Aggregation Network for Image Super-Resolution, CVPR, 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.html)

[75] Jae Woong Soh, Sunwoo Cho, Nam Ik Cho. Meta-Transfer Learning for Zero-Shot Super-Resolution. CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Soh_Meta-Transfer_Learning_for_Zero-Shot_Super-Resolution_CVPR_2020_paper.html)

[76] Yong Guo, Jian Chen, Jingdong Wang, Qi Chen, Jiezhang Cao, Zeshuai Deng, Yanwu Xu, Mingkui Tan. Closed-Loop Matters: Dual Regression Networks for Single Image Super-Resolution. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Closed-Loop_Matters_Dual_Regression_Networks_for_Single_Image_Super-Resolution_CVPR_2020_paper.html)

[77] Yiqun Mei, Yuchen Fan, Yuqian Zhou, Lichao Huang, Thomas S. Huang, Honghui Shi. Image Super-Resolution With Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining. CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Mei_Image_Super-Resolution_With_Cross-Scale_Non-Local_Attention_and_Exhaustive_Self-Exemplars_Mining_CVPR_2020_paper.html)

[78] Fuzhi Yang, Huan Yang, Jianlong Fu, Hongtao Lu, Baining Guo. Learning Texture Transformer Network for Image Super-Resolution, CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Learning_Texture_Transformer_Network_for_Image_Super-Resolution_CVPR_2020_paper.html)

[79] Jaejun Yoo, Namhyuk Ahn, Kyung-Ah Sohn. Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy. CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Yoo_Rethinking_Data_Augmentation_for_Image_Super-resolution_A_Comprehensive_Analysis_and_CVPR_2020_paper.html)

[80] Gyumin Shim, Jinsun Park, In So Kweon. Robust Reference-Based Super-Resolution With Similarity-Aware Deformable Convolution, CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.html)

[81] Yu-Syuan Xu, Shou-Yao Roy Tseng, Yu Tseng, Hsien-Kai Kuo, Yi-Min Tsai. Unified Dynamic Convolutional Network for Super-Resolution With Variational Degradations, CVPR 2020. [[Paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Unified_Dynamic_Convolutional_Network_for_Super-Resolution_With_Variational_Degradations_CVPR_2020_paper.html)

[82] Yanchun Xie, Jimin Xiao, Mingjie Sun, Chao Yao, Kaizhu Huang. Feature Representation Matters: End-to-End Learning for Reference-based Image Super-resolution. ECCV, 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490222.pdf)

[83] Andreas Lugmayr, Martin Danelljan, Luc Van Gool, Radu Timofte. SRFlow: Learning the Super-Resolution Space with Normalizing Flow. ECCV, 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500698.pdf)

[84] Yulun Zhang, Zhifei Zhang, Stephen DiVerdi, Zhaowen Wang, Jose Echevarria, Yun Fu. Texture Hallucination for Large-Factor Painting
Super-Resolution. ECCV, 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520205.pdf)

[85] Xiaotong Luo, Yuan Xie, Yulun Zhang, Yanyun Qu, Cuihua Li, Yun Fu. LatticeNet: Towards Lightweight Image Super-resolution with Lattice Block. ECCV, 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670273.pdf)

[86] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, Liang Lin. Component Divide-and-Conquer for Real-World Image Super-Resolution. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530103.pdf)

[87] Ben Niu, Weilei Wen, Wenqi Ren, Xiangde Zhang, Lianping Yang, Shuzhen Wang, Kaihao Zhang, Xiaochun Cao, Haifeng Shen. Single Image Super-Resolution via a Holistic Attention Network. ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570188.pdf)

[88] Majed El Helou, Ruofan Zhou, Sabine Süsstrunk. Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks. ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610732.pdf)

[89] Xi Cheng, Zhenyong Fu, Jian Yang. Stochastic Frequency Masking to Improve Super-Resolution and Denoising Networks. ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf)

[90] Sangeek Hyun, Jae-Pil Heo. VarSR: Variational Super-Resolution Network for Very Low Resolution Images. ECCV, 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680426.pdf)

[91] Wonkyung Lee, Junghyup Lee, Dohyung Kim, Bumsub Ham. Learning with Privileged Information for Efficient Image Super-Resolution. ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690460.pdf)

[92] Huixia Li, Chenqian Yan, Shaohui Lin, Xiawu Zheng, Baochang Zhang, Fan Yang, Rongrong Ji. PAMS: Quantized Super-Resolution via Parameterized Max Scale. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700562.pdf)

[93] Royson Lee, Łukasz Dudziak, Mohamed Abdelfattah, Stylianos I. Venieris, Hyeji Kim, Hongkai Wen, Nicholas D. Lane. Journey Towards Tiny Perceptual Super-Resolution. ECCV 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710086.pdf)

[94] Seobin Park, Jinsu Yoo, Donghyeon Cho, Jiwon Kim, Tae Hyun Kim. Fast Adaptation to Super-Resolution Networks via Meta-Learning. ECCV, 2020. [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710086.pdfx)

[95] Zhengxiong Luo, Yan Huang, Shang Li, Liang Wang, Tieniu Tan. Unfolding the Alternating Optimization for Blind Super Resolution. NeurIPS, 2020. [[Paper]](https://arxiv.org/pdf/2010.02631.pdf) [[Code]](https://github.com/greatlog/DAN)


