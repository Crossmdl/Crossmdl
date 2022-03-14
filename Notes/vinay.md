# Cross Modal Retrieval and Alignment.

The Option 2 task requires aligning text with instructional videos. 

`CURRENT THOUGTS`: Use features extraction similar to DROP-DTW for both video and text.(as it's trained to optimize text-video similarity). Now formulate the problem as aligning video-text pairs. Now Youcook2 also has acoustic data. Try to utilize it(multimodal transformer etc).

Some useful resources:

[Overview of object detection literature](https://www.youtube.com/watch?v=her4_rzx09o)

[RCNN review slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/FA2020/598_FA2020_lecture15.pdf)

`Input at inference`:

a. Video.
b. Instructional text

`Input at Training time`:

Same as above + (matched correspondance between video and text.)

`Output`:

Match the frame of the video to a input text-sentence. 

`Comments:`
Consider N-video frames and M-sentences. There exist a potentially many-one mapping between the two - many video frame correspond to a single sentence. Moreover many video frames do not belong to any sentence. Key points:

a. The input video length is fixed.(we sample the video to fixed frames say 512)

b. But potential text-sentences are varied.(with atmost the video length,thus each frame correponding to a different text token)


`Formulation 2`:
From the training videos, we
find that 7 temporal window lengths typically exist: 25, 60,
78, 100, 150, 190 and 250 frames. We then fix a sliding
step size of 10 frames. Finally, we perform non-maximum
suppression to ignore overlapping detection windows.

by computing similarity be-tween every frame in the video and the action label namesof CrossTask.


`metrics`
mAP score over all activity classes. To do
this, a detection is determined to be a true positive according to the following procedure: 1) we compute the overlap
(measured by the intersection over union score) between a
predicted temporal segment and a ground truth segment, 2)
we mark the detection as positive if the overlap is greater
than a threshold ↵. In practice, we vary the threshold ↵
between 0.1 and 0.5.

We evaluate our learned embedding using the stan-dard recall metrics R@1, R@5, R@10 and the median rank(Median R).

Drop-DTW Paper used the pretrained [Youtube100M dataset](chrome-extension://dagcmkpagjlhakfdhnbomgmjdpkdklff/enhanced-reader.html?openApp&pdf=https%3A%2F%2Farxiv.org%2Fpdf%2F1912.06430.pdf)

[code/weights](https://github.com/antoine77340/S3D_HowTo100M)

Also the [Drop-DTW paper code](https://github.com/SamsungLabs/Drop-DTW/tree/64a64dff20ee8b3cfdb7edb3e793a5a265af25fd)



`Modeling`

[Approach 1](https://arxiv.org/abs/1803.00057): Baseline

Given a video and a text-token output should have [(center(0,1->scale invariant),width),can have different parametrizations.],label = 0,1(to indicate if whether that sentence belongs to the video or not).

Note that here the output's (location) of sentences are non-overlapping. It would be better to have a penalty during training to have the output regions non-overlapping. Can penalize by the (intersection/total_length) of proposed intervals.

# Approach 2(adapt from the following)

### Multi-modal Transformer for Video Retrieval(MMT)

Input : only text narration but no time-annotation.

`Multi-modal Transformer for Video Retrieval`(mmt)

Self-attention(with modality encoding) to aggregate multimodal information.[1](https://www.mendeley.com/reference-manager/reader/e6e62e7a-8730-3088-8513-a3b9598b8d39/42c48f26-4c32-9f83-7639-3e4e646df76b). They trained on video-text simililarity.Their [code](https://github.com/gabeur/mmt). Another approach is using Cross-attention as in [2](https://www.mendeley.com/reference-manager/reader/3c914788-db70-3233-ab06-820dc11c4254/5d7262e4-013b-9be1-1b0f-328b4ab4b7eb),[Code - easy and readable](https://github.com/yaohungt/Multimodal-Transformer/tree/a670936824ee722c8494fd98d204977a1d663c7a).  

Summary:

<img width="700" alt="Screen Shot 2022-03-11 at 9 59 00 PM" src="https://user-images.githubusercontent.com/21222766/158001266-130779b7-b81c-4b8b-9568-47e25fb2f528.png">

<img width="700" alt="Screen Shot 2022-03-11 at 9 59 15 PM" src="https://user-images.githubusercontent.com/21222766/158001272-4dd338a6-01c4-4b01-a88a-a99ab7beabef.png">



[Learning a Text-Video Embedding from Incomplete and Heterogeneous Data](https://arxiv.org/abs/1804.02516)

This paper is important. Read.




Can we use pretarining ?(further paper suggestions)

1.[Integrating Multimodal Information in Large Pretrained Transformers](https://arxiv.org/pdf/1908.05787.pdf)


Just modify the input embedding vectors of text based on other modalities.Comment: Check their code to see how they adapted BERT.

<img width="432" alt="Screen Shot 2022-03-13 at 10 14 20 PM" src="https://user-images.githubusercontent.com/21222766/158093352-16d6bacf-5199-4bf4-a32f-aad36b6892a7.png">

<img width="416" alt="Screen Shot 2022-03-13 at 10 14 44 PM" src="https://user-images.githubusercontent.com/21222766/158093372-029848e8-79cb-4f11-8fb4-92b721787a01.png">

<img width="469" alt="Screen Shot 2022-03-13 at 10 15 11 PM" src="https://user-images.githubusercontent.com/21222766/158093385-23369c26-ab83-4942-9285-59c1e479a582.png">


2.[VLBERT](https://arxiv.org/pdf/1908.08530.pdf) - similar to MMT but they released their pretrained model. You can finetune.

3.[LXMERT: Learning Cross-Modality Encoder Representations
from Transformers](https://piazza.com/class_profile/get_resource/kcnr11wq24q6z7/kfmv96gl6at6gg)

4.[Multimodal Transformer Networks for End-to-End
Video-Grounded Dialogue Systems](https://piazza.com/class_profile/get_resource/kcnr11wq24q6z7/kfmv97aseti6i1)

5.Cross-Modality Relevance for Reasoning on Language and Vision(https://arxiv.org/abs/2005.06035)


************
PYTORCH LIGHTNING AND Deep Learning [tutorial series](https://uvadlc-notebooks.readthedocs.io/en/latest/)


