# Weeds Detection and Segmentation
This repository contains the code for my OUR(Opportunities for Undergraduate Research) project at Shiv Nadar University under the supervision of Prof. Jyoti Singh Kirar, Professor at Dept. of Computer Science, Shiv Nadar University. 

You can view the final project report here: [Final Report](https://docs.google.com/document/d/1yffPnYz8fYgYJVYeD-dMnylnXalV5GknPbtPAAnE6ro/edit#heading=h.nj23sjpj5u97)

---

# The Dataset
I used the [Open Sprayer Images Dataset](https://www.kaggle.com/gavinarmstrong/open-sprayer-images) on Kaggle for the project. I used this dataset with some data augmentation for the purpose of Weeds Classification.
For the Weeds Segmentation task, I created a custom dataset using the same dataset by hand painting binary masks using an Gimp - An image manipulation tool. You can download this dataset [here](https://drive.google.com/open?id=1vbw6itGLk59haxVBjNlXsnOmAbbwlAZy)

---

# Implementing the Code

* Download the [Open Sprayer Images Dataset](https://www.kaggle.com/gavinarmstrong/open-sprayer-images) and rename it to 'Open_Sprayer_Images_Classification' and add a folder named 'docks_augmented' inside that folder.
* Download the [Segmentation Dataset](https://drive.google.com/open?id=1vbw6itGLk59haxVBjNlXsnOmAbbwlAZy) and place it inside the 'Dataset' folder.
* Download the [Trained Weights](https://drive.google.com/open?id=1tjwl0oQxIAkUF_08Qa37WDA3Dc1HtV7x) for the CNN and place it in the 'weights' folder in the 'Code' folder.
* Download the [Trained Models](https://drive.google.com/open?id=142H87gEpVTG3RgJWGKnstCG5531xDoCF) and place it in the 'models' folder in the 'Code' folder.
* Run the Jupyter Notebook in the 'Code' Folder

---

# References
1. H. C. Oliveira, V. C. Guizilini, I. P. Nunes and J. R. Souza, "Failure Detection in Row Crops From UAV Images Using Morphological Operators," in IEEE Geoscience and Remote Sensing Letters, vol. 15, no. 7, pp. 991-995, July 2018.
2. S. Kaur, S. Pandey and S. Goel, "Semi-automatic leaf disease detection and classification system for soybean culture," in IET Image Processing, vol. 12, no. 6, pp. 1038-1048, 6 2018.
3. Bah, Mamadou & Hafiane, Adel & Canals, R. (2018). Deep Learning with Unsupervised Data Labeling for Weed Detection in Line Crops in UAV Images. Remote Sensing. 10. 10.3390/rs10111690. 
4. Sa, I., Popovic, M., Khanna, R., Chen, Z., Lottes, P., Liebisch, F., Nieto, J.I., Stachniss, C., Walter, A., & Siegwart, R. (2018). WeedMap: A Large-Scale Semantic Weed Mapping Framework Using Aerial Multispectral Imaging and Deep Neural Network for Precision Farming. Remote Sensing, 10, 1423.
5. Abdullahi, Halimatu & Sheriff, Ray & Mahieddine, Fatima. (2017). Convolution neural network in precision agriculture for plant image recognition and classification. 1-3. 10.1109/INTECH.2017.8102436. 
6. Chen, Liang-Chieh, et al. ”Semantic image segmentation with deep convolutional nets and fully connected crfs.” arXiv preprint arXiv: 1412.7062(2014).
7. Liu, Baiting and Yeming Wen. “CNN and CRF for Semantic Image Segmentation.” (2016).
8. Huang H, Deng J, Lan Y, Yang A, Deng X, et al. (2018) A fully convolutional network for weed mapping of unmanned aerial vehicle (UAV) imagery. PLOS ONE 13(4): e0196302. 
9. Herrera, P. Javier et al. “A Novel Approach for Weed Type Classification Based on Shape Descriptors and a Fuzzy Decision-Making Method.” Sensors (2014).
10. Dyrmann, M. (2017). Automatic Detection and Classification of Weed Seedlings under Natural Light Conditions. Syddansk Universitet. Det Tekniske Fakultet.
11. Ronneberger, Olaf et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” MICCAI (2015).
12. Sa, Inkyu et al. “weedNet: Dense Semantic Weed Classification Using Multispectral Images and MAV for Smart Farming.” IEEE Robotics and Automation Letters 3 (2018): 588-595.
13. Kodagodaa, S. et al. “Weed detection and classification for autonomous farming.” (2013).
14. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. CoRR, abs/1409.1556.
15. V. Vapnik. The Nature of Statistical Learning Theory. Springer-Verlag, 2nd edition, 1998.
16. Krahenbuhl, P. & Koltun, V. (2011). Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials. Neural Information Processing Systems. 
17. Fradkin, Dmitriy & Muchnik, Ilya. (2019). Support Vector Machines for Classification. 
18. A. S. More and D. P. Rana, "Review of random forest classification techniques to resolve data imbalance," 2017 1st International Conference on Intelligent Systems and Information Management (ICISIM), Aurangabad, 2017, pp. 72-78.doi: 0.1109/ICISIM.2017.8122151
19. Khaled Fawagreh, Mohamed Medhat Gaber & Eyad Elyan (2014) Random forests: from early developments to recent advancements, Systems Science & Control Engineering, 2:1, 602-609, DOI: 10.1080/21642583.2014.956265
20. Breiman (2001) Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. doi: 10.1023/A:1010933404324
21. Ho, T. K. (1995). Random decision forests. In Document analysis and recognition, 1995, Proceedings of the third international conference, Montreal, Quebec, Canada (Vol. 1, pp. 278–282). New York City, NY: IEEE. [Google Scholar]
22. Ho (1998) Ho, T. K. (1998).The random subspace method for constructing decision forests. Intelligence, IEEE Transactions on Pattern Analysis and Machine, 20(8), 832–844. doi: 10.1109/34.709601 [Crossref], [Web of Science ®], , [Google Scholar] 
23. Amit and Geman (1997) Amit Y., & Geman, D. (1997). Shape quantization and recognition with randomized trees. Neural Computation, 9(7), 1545–1588. Doi: 10.1162/neco.1997.9.7.1545.[Crossref], [Web of Science ®], [Google Scholar]
24. Sharma, Himani & Kumar, Sunil. (2016). A Survey on Decision Tree Algorithms of Classification in Data Mining. International Journal of Science and Research (IJSR). 5.
25. A. Arnab et al., "Conditional Random Fields Meet Deep Neural Networks for Semantic Segmentation: Combining Probabilistic Graphical Models with Deep Learning for Structured Prediction," in IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 37-52, Jan. 2018. doi: 10.1109/MSP.2017.2762355
26. Tian, L.; Reid, J.F.; Hummel, J.W. Development of a precision sprayer for site-specific weed management. Trans. Am. Soc. Agric. Eng. 1999, 42, 893–900.
27. Timmermann, C.; Gerhards, R.; Kühbauch, W. The economic impact of site-specific weed control. Precis. Agric. 2003, 4, 249–260.
28. Gerhards, R.; Oebel, H. Practical experiences with a system for site-specific weed control in arable crops using real-time image analysis and GPS-controlled patch spraying. Weed Res. 2006, 46, 185–193.
29. Nordmeyer, H. Patchy weed distribution and site-specific weed control in winter cereals. Precis. Agric. 2006, 7, 219–231.
30. Kumar, D., Verma, H., Mehra, A., & Agrawal, R.K. (2018). A modified intuitionistic fuzzy c-means clustering approach to segment human brain MRI image. Multimedia Tools and Applications, 1-25.

