# Multimodal Action Recognition

In this reproducibility project, we are reproducing a subset of experiments presented in the paper submission to ICLR 2019 titled _Unseen Action Recognition with Multimodal Learning_. The authors leverage a deep autoencoder model to learn a joint multimodal representation over video and textual data. To obtain the joint representations, they train two separate autoencoder models one using video data and another with corresponding caption data, as proposed by Ngiam et al.<sup>[1](#myfootnote1)</sup>


The authors feed the video data to a pre-trained _Two-Stream Inflated 3D ConvNet_ (I3D) <sup>[2](#myfootnote2)</sup> to obtain an initial I3D video representation. The resulting I3D representation is then used as the input to an autoencoder network with temporal attention filters<sup>[3](#myfootnote3)</sup> to learn video embeddings. The textual data, on the other hand, is represented as GloVe<sup>[4](#myfootnote4)</sup>  word embeddings and are used to train another autoencoder network with a similar architecture. The authors train the model and evaluate it in two different setups: (i) using _paired_ video and textual data; and (ii) using both _paired_ and _unpaired_ video and textual data.

We work to reproduce the results obtained from the authors' proposed architecture in multiple settings; to validate authors justification for their choice of network architecture, the loss functions, as well as their evaluation methodology. We use the ActivityNet video<sup>[5](#myfootnote5)</sup> and captions <sup>[6](#myfootnote6)</sup> datasets in our experiments and we focus on the case with paired video and captions. 

Our code is implemented in _pytorch_. The process of training the model on ActivityNet dataset is highly resource-intensive. Although the code-base is ready to perform the proposed experiments; we could not train the model with the low resources available on the VM that was assigned to this project. The experiments are due to run on _Calcul Québec_ cluster.

---

<a name="myfootnote1">1</a>:  Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, and Andrew Y Ng. Multimodal Deep Learning. In The 28th International Conference on Machine Learning (ICML), 2011.

<a name="myfootnote2">2</a>: Karen Simonyan and Andrew Zisserman. Two-Stream Convolutional Networks for Action Recognition in Videos. In Advances in Neural Information Processing Systems 27 (NIPS), 2014.

<a name="myfootnote3">3</a>:  AJ Piergiovanni, Chenyou Fan, and Michael S Ryoo. Learning latent sub-events in activity videos using temporal attention filters. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017.

<a name="myfootnote4">4</a>: Jeffrey Pennington, Richard Socher, and Christopher D Manning. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.

<a name="myfootnote5">5</a>: Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 961–970, 2015.

<a name="myfootnote6">6</a>: Ranjay Krishna, Kenji Hata, Frederic Ren, Michael Bernstein, Fei-Fei Li, and Juan Carlos Niebles. Dense-captioning events in videos. In ArXiv , 2017.

