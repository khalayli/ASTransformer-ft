# ASTransformer-ft

## Description
Originally part of a university Deep Learning course project, the ASTransformer-ft represents my endeavor to tackle the intricacies of Music Genre Classification using Audio Spectogram Transformers. Initially dissatisfied with the results, I revisited the project four months later, leading to several iterations and refinements.

## Related Work: Music Genre Classification Research Summary
This section outlines a comprehensive summary of various research studies in music genre classification, spotlighting datasets used, architectures employed, accuracies achieved, and observed weaknesses.

### Research Papers Overview
A table summarizing key aspects of different research papers on music genre classification, highlighting datasets, architectures, accuracies, and notable comments or weaknesses.

| Paper Reference | Dataset Used                | Architecture Used                              | Best Accuracy Achieved | Notes / Weaknesses                                         |
|-----------------|-----------------------------|------------------------------------------------|------------------------|------------------------------------------------------------|
| Paper 2         | GTZAN dataset               | ResNet-18                                      | 82.71%                 | Annotation errors, limited samples                         |
| Paper 3         | FMA dataset                 | Ensemble of LSTM and CNN architectures         | 89.25%                 | Taxonomy inconsistencies, potential overfitting            |
| Paper 4         | GTZAN dataset               | ANN 2, CNN (Conv-Conv-Pool)                    | 94.1% for ANN 2        | Limited to 30-second song fragments                        |
| Paper 5         | Large database (COMAP)      | BP neural network                              | 91.4% on testing set   | Uneven distribution of samples across genres               |
| Paper 6         | GTZAN dataset               | Parallelized CNN attention                     | 91.4% on testing set   | Limited size, potential for overfitting                    |
| Paper 7         | GTZAN dataset               | Bidirectional gated recurrent units (BI-GRU)   | 90.1% on validation set| Limited focus on MIDI music; May not cover all subgenres   |
| Paper 16        | GTZAN dataset               | AST model with input-dependent NMR             | 82.8%                  | Utilizes pre-trained models for low-resource settings      |
| Paper 23        | GTZAN dataset with 10 genres| Custom CNN model                               | 98.30%                 | Recognizes only ten predefined genres                      |
| Paper 30        | Indonesian regional songs   | Deep Recurrent Neural Network (DRNN)           | 83.28%                 | Limited explanation of data collection process             |

References to the papers are included in the Zotero file in the repo.
This table presents a concise overview of the diverse approaches and findings within the realm of music genre classification research, providing insights into the methodologies and outcomes of recent studies.


### Accuracies Observed in Previous Studies
This subsection details accuracies reported in various studies, shedding light on the performance of different approaches in music genre classification.

Sigtia and Dixon (2014): Representation learning with an accuracy of 74.4%.

Costa et al. (2017): Visual features (LBP) achieving an accuracy of 80.6%.

Nanni et al.: Visual features (LPB-HF, LBP, RICLBP, LPQ) with an accuracy of 82.9%.

Seyerlehner et al.: Acoustic features with an accuracy of 88.3%.

Lim et al. (2012): Acoustic features reaching an accuracy of 89.9%.

Nanni et al.: A combination of visual (LPB-HF, LBP, RICLBP) and acoustic features (DFB, OSC) for an accuracy of 90.2%.

**Costa et al. (2017) also reported an accuracy of 85.9% with CNN for music classification using spectrograms on the ISMIR 2004 dataset.**

### Why Transformers?

The use of transformers in audio genre classification, based on insights from various studies, offers several compelling advantages that make them particularly suitable for this application. Although the specific papers provided earlier were not explicitly tied to detailed discussions on transformers, the general advantages of using transformer models in tasks similar to audio genre classification can be inferred from current advancements in neural network research and application. Here are reasons to why I chose transformers for this audio genre classification project, inspired by broader research findings and the capabilities of transformer models:


Attention Mechanism: They focus on the most relevant parts of an audio clip, identifying distinctive musical features crucial for genre classification.

Sequential Data Handling: Unlike RNNs, transformers capture long-range dependencies in audio data efficiently, understanding complex temporal patterns.

Parallel Processing: Their ability to process data in parallel enables handling large datasets effectively, crucial for extensive music collections.

Automatic Feature Extraction: Transformers reduce the need for manual feature engineering by learning from a wide range of data features directly.

Adaptability: The architecture is flexible, allowing for fine-tuning on diverse audio features and accommodating various audio analysis tasks.

Performance: They often outperform other models in complex classification tasks by better distinguishing between genres with subtle differences.

Multimodal Integration: Transformers can incorporate additional data types, such as lyrics, enhancing classification accuracy through a holistic approach to music analysis.

These attributes make transformers a powerful choice for improving accuracy and efficiency in audio genre classification.


### Dataset Used: ISMIR2004 Genre Dataset
The ISMIR2004 Genre Dataset, stemming from the Genre Identification task of the ISMIR 2004 audio description contest, serves as the cornerstone of this project. It encompasses audio tracks across 8 genres, organized for classification into 6 classes.

The dataset is comprised of audio tracks encoded in MP3 format, organized into three distinct folders to represent various subsets of the entire collection. Each of these folders contains 729 files, categorized into classes that mirror the genre distribution present in Magnatune at the time of the dataset's creation. It's important to note that each track is unique to its folder, ensuring no duplicates across the dataset.

Training Set: This folder is designated for the creation of classification models, with files methodically sorted by genre.
Development Set: Intended for model validation, this folder contains a distinct set of tracks that allows participants to test and refine their models.
Evaluation Set: Initially kept private, this subset includes tracks specifically used to assess the performance of models submitted for evaluation.
The composition of the Training and Development sets is as follows:

Classical: 320 files

Electronic: 115 files

Jazz & Blues: 26 files

Metal & Punk: 45 files

Rock & Pop: 101 files

World: 122 files

The Evaluation set, mirroring a similar genre distribution, encompasses 729 tracks, maintaining consistency across the dataset.

## Challenges and Solutions
TorchAudio created lots of problems from start to beginning and had major memory management issues leading to many crashed, it was a last resort to use liberosa for preprocessing and audio loading, it is 30x slower but works. 

The nature of the ismir2004 audio dataset being mp3 files made it a little more difficult to process as not all libraries natively support mp3 but it was also one of the reasons i insisted to continue using it.

Converting to a huggingface dataset for Trainer was also a complication because it was not straightforward and required lots of work due to the dataset's mp3 nature.

## Results
The model demonstrates a strong capacity for music genre classification, as evidenced by the results obtained from the evaluation of the ISMIR 2004 Genre Dataset. The model achieved an accuracy of 86.15%, along with precision, recall, and F1 scores that indicate a balanced performance across different genres. The loss at evaluation was measured at 0.6254.


## ROC Curve

![Unknown](https://github.com/khalayli/ASTransformer-ft/assets/154463029/77a6a0aa-fc26-4100-bbe5-4040ea962fd0)

The multi-class ROC curve illustrates the model's true positive rate against the false positive rate for each genre, providing an understanding of the trade-offs between sensitivity and specificity:

Classical music achieved a perfect area under the curve (AUC) of 1.00, showcasing the model's exceptional performance in this genre.

All genres performed well, with AUC scores ranging from 0.94 to 0.98, indicative of the model's overall effective classification capability.

The results are promising, and they reflect the model's proficiency in distinguishing between different musical elements characteristic of various genres. While the performance on Jazz & Blues and Rock & Pop indicates potential areas for refinement, the overall effectiveness of the model is evident through the high accuracy and AUC scores achieved.

## Confusion Matrix 

![Unknown-3](https://github.com/khalayli/ASTransformer-ft/assets/154463029/726f7995-222f-4f63-a720-7d112421ab6b)

The confusion matrix presents a clear visualization of the model's performance, with the following highlights:

Classical tracks were exceptionally well-classified, with a true positive rate of 312 out of 320 instances, corroborating the model's ability to discern features unique to classical music.

Electronic and World genres also saw high accuracy, although there were some instances of confusion between the two, possibly due to overlapping musical elements.

Rock & Pop and Metal & Punk genres were more challenging for the model, with the greatest number of misclassifications occurring between these classes and some rock & pop being classified as world.

## Normalized Confusion Matrix

![Unknown-2](https://github.com/khalayli/ASTransformer-ft/assets/154463029/08bd5d5f-ca5f-4533-8590-cacb0a580eee)

The normalized confusion matrix further refines our understanding:

Classical genre classification is nearly perfect with a score of 0.99, signifying a robust identification of this genre.

Electronic stands strong at 0.89, though it seems to be occasionally confused with world and Rock & Pop.

Metal & Punk has the lowest diagonal value of 0.60, indicating room for improvement, especially distinguishing from Rock & Pop.

The other genres show a fair level of accuracy with some room for improvement in cross-genre differentiation.





## Reflections and Future Work
It was a difficult but interesting project which took a great time and effort and most importantly commitment (more than 20 iterations) but I'm satisified with the results. Future improvements would include class weights for the underrepresented class weights, synthetic data etc.. 

## Acknowledgements

This project has benefited greatly from the use of external resources and datasets, which have been instrumental in the development and refinement of the ASTransformer-ft. Special thanks are extended to:

- The ISMIR2004 Genre Dataset, derived from the Genre Identification task of the ISMIR 2004 audio description contest, organized by the Music Technology Group (Universitat Pompeu Fabra). This dataset played a pivotal role in the genre classification tasks of this project. Further details about the dataset and contest can be found [here](http://ismir2004.ismir.net/genre_contest/).

- The AST model fine-tuned on AudioSet by MIT, available on Hugging Face at [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

## Licensing
The audio in the ISMIR2004 Genre Identification task dataset is licensed under a CC Attribution-NonCommercial-ShareAlike license. Please adhere to this license when using the dataset.

## Contact
Sami Khalayli | s.khalayli12@gmail.com | [linkedin.com/in/khalayli](https://www.linkedin.com/in/khalayli/)


