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
>[FACS](https://pdfs.semanticscholar.org/99bf/8ac8c131291d771923d861b188510194615e.pdf), AU, [Prior probobility](https://bayes.wustl.edu/etj/articles/prior.pdf), [FACS codes](https://www.cs.cmu.edu/~face/facs.htm), [PCA](https://medium.com/@aptrishu/understanding-principle-component-analysis-e32be0253ef0), [Eigenvector](http://setosa.io/ev/eigenvectors-and-eigenvalues/), [Marginal/Conditional Probability](http://tinyheero.github.io/2016/03/20/basic-prob.html), [Generative Classifiers VS Discriminative Classifiers](http://www.chioka.in/explain-to-me-generative-classifiers-vs-discriminative-classifiers/), [Bayes Theorem](http://www.chioka.in/explain-to-me-bayes-theorem/)

    * Challenges identified
        * Lack of enough AU annotated dataset (difficult for FER)
        * Models using AU relationships still need AU annotated data or expression labelled
    * How is porposed solution better
        * utilizes prior probablities than depending on annotated AUs
        * developed algorithm to optimize formulated problem by iteratively updating classifiers and AU labels
        * Demonstrates accuracy on 5 benchmark datasets compared to SoA
    * Technical explanation
        * Generative Models (Bayes Network, RBM) and Discriminative Models (EmotionNet) have attempted Multi-labal AU classification but at best even they depend on annotated AU samples. They utilize AU<=>AU & AU<=>Expression relations.
        * Ruiz et al proposed model is not dependent on annotated AUs but needs expression labelled samples
        * Unlike Ruiz's model, current approach utilizes individual and joint probablities. Also can handle exact probablities along with inequality
        * Prior probablity has been classified into expression independed/dependent category
        * AUs are classified into primary, secondary and others based on their role in the facial expression
    * Need to re-read optimization section
    * Didn't understand loss function definition

$$P(y_i=1 | y_j=1) > P\left(y_i=0 | y_j=1\right)$$
Applying bayes theorem to either side of inequality
$$P(y_i=1 | y_j=1) = {{P(y_j=1 | y_i=1)P(y_j=1)} \over {P(y_i=1)}}$$
following can be derived
$$P(y_j=1 | y_i=1) > P(y_j=1)$$

3. [Joint Action Unit localisation and intensity estimation through heatmap regression](https://arxiv.org/abs/1805.03487)
>heatmap, [code](https://github.com/ESanchezLozano/Action-Units-Heatmaps), Hourglass architecture

    * Challenges identified
        * Models sensitive to facial landmarks fail due to error
        * Complex DNN work on fast-gpu but cannot work on portable devices due to limited compute
    * How is porposed solution better
        * jointly AU localisation and intensity estimation, where AU is modelled  with heatmap
        * Low complexity solution - heatmap regression with single hourglass network
        * Demonstrated robustness to misalignment errors
        * heatmap regression allows to learn from shared representation between AUs without the need of latent representations as it's implicitly learned from data
    * Technical explanation
        * TBD

4. [Unsupervised Features for Facial Expression Intensity Estimation over Time](https://arxiv.org/pdf/1805.00780.pdf)
5. [Expression Empowered ResiDen Network for Facial Action Unit Detection](https://arxiv.org/pdf/1806.04957.pdf)
>DISFA, EmotionNet, RAF-DB

Automated FER
>is active research problem either catering to six prototypical emotions or AU (set of atomic faciacl muscles actions from which any expression be inferred) recognition. It is generally treated as Multi-label classifier or regression problem

FACS (facial action coding system)
>is a protocol used to objectively identify facial expression,without relation to emotion, caused by specific facial muscles. Action unit is one such smallest discrminiable facial expression ,creating a change in facial appearance, & measured in frequency/intensity. Intensity can range from trace-level=>slight=>marked=>pronounced=>severe=>extreme=>maximum

*  Face - 27, Head and eye position - 25, Misc (show tongue) - 28
*  Applying FACS is time-consuming & reliability is tougher depending on experience of FACS coder
*  FACS in Infant is a separate study
*  Emotion expression in FACS - that liars who were smiling were also likely to have short-lived, trace-level Action Units frequently associated with disgust, sadness, or anger
*  Eye constriction -  proposed that smiles that included eye constriction (also known as Duchenne smiles) indexed strong positive emotion
*  Automated measurment of FACS - AAM (active appearance models) extracts and represent shape and appearance features from a video sequence thereafter SVM (support vector machines) classifiers detect AU & quantify their intensity.

Prior Probablity (need to read more)
>In decision theory, mathematical analysis has demostrated that once sample distribution, loss function and samples are defined only remaining basis for choice among different admissible decisions lies in prior probability.

* Orthodox staticians rejected the idea of prior probability since it couldn't be mathematically proven with then available methods except when it consists of frequency data
* However, with advent of modern decision theory prior probabilityhas gain ground
* Since prior information may not be completely measurable hence "in two problems where we have the same prior information, we should assign the same prior probabilities" 

Search links
* [link1](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=%22action+unit%22&terms-0-field=all&classification-physics_archives=all&date-year=&date-filter_by=date_range&date-from_date=2018-01&date-to_date=2018-07&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first)
* [link2](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=%22Emotion+Recognition%22&terms-0-field=all&terms-1-operator=NOT&terms-1-term=%22speech%22&terms-1-field=all&classification-physics_archives=all&date-year=&date-filter_by=date_range&date-from_date=2018-01&date-to_date=2018-07&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first)

