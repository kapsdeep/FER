### Paper reading for FER

1. [Transfer Learning for Action Unit Recognition ](https://arxiv.org/pdf/1807.07556.pdf)
>FACS, AU, FER, LDA, LSTM, SVM, VGG-Face, ResNet, EMFACS, DISFA, ENSEMBLE, F1 score, classification rate

    * Challenges identified
        * DNNs require large set of annotated dataset (difficult for FER)
        * DNNs require high compute power => expensive gpu
        * DNNs require long training time in weeks/months
    * How is porposed solution better
        * recognising occurrence of 12 AUs using transfer learning
        * Ensemble of VGG-Face & ResNet
        * Feature extraction
    * Technical explanation
        * AU recognition performance ascending order: VGGNet=>ResNet=>Ensemble of region based CNNs
        * In this project, we extract features from the first fully connected layer after convolutional blocks
    * Topics index
        * Introduction
        * CNN Architectures (describing alexnet, googlenet...)
        * Database (Analysing DISFA)
        * Feature Extraction
            * Pre-processing => Use CNNs to extract features => Train linear classifiers => LSTM for temporal info => Region-based approach (segment face into 3 regions)
        * Fine-tuning
        * Classifier ensemble
        * Results
2. [Classifier Learning with Prior Probabilities for Facial Action Unit Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Classifier_Learning_With_CVPR_2018_paper.pdf)








Search links
* [link1](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=%22Emotion+Recognition%22&terms-0-field=all&terms-1-operator=NOT&terms-1-term=%22speech%22&terms-1-field=all&classification-physics_archives=all&date-year=&date-filter_by=date_range&date-from_date=2018-01&date-to_date=2018-07&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first)
* [link2](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=%22action+unit%22&terms-0-field=all&classification-physics_archives=all&date-year=&date-filter_by=date_range&date-from_date=2018-01&date-to_date=2018-07&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first)
