# Cross Modal Retrieval and Alignment.

The Option 2 task requires aligning text with instructional videos. 

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

`Modeling`

[Approach 1](https://arxiv.org/abs/1803.00057):

Given a video and a text-token output should have [(center(0,1->scale invariant),width),can have different parametrizations.],label = 0,1(to indicate if whether that sentence belongs to the video or not).

Note that here the output's (location) of sentences are non-overlapping. It would be better to have a penalty during training to have the output regions non-overlapping. Can penalize by the (intersection/total_length) of proposed intervals.

# Paper Summary

### Multi-modal Transformer for Video Retrieval

Input : only text narration but no time-annotation.

`Multi-modal Transformer for Video Retrieval`

Self-attention(with modality encoding) to aggregate multimodal information.[1](https://www.mendeley.com/reference-manager/reader/e6e62e7a-8730-3088-8513-a3b9598b8d39/42c48f26-4c32-9f83-7639-3e4e646df76b). They trained on video-text simililarity.Their [code](https://github.com/gabeur/mmt). Another approach is using Cross-attention as in [2](https://www.mendeley.com/reference-manager/reader/3c914788-db70-3233-ab06-820dc11c4254/5d7262e4-013b-9be1-1b0f-328b4ab4b7eb),[Code - easy and readable](https://github.com/yaohungt/Multimodal-Transformer/tree/a670936824ee722c8494fd98d204977a1d663c7a).  

Summary:

<img width="700" alt="Screen Shot 2022-03-11 at 9 59 00 PM" src="https://user-images.githubusercontent.com/21222766/158001266-130779b7-b81c-4b8b-9568-47e25fb2f528.png">

<img width="700" alt="Screen Shot 2022-03-11 at 9 59 15 PM" src="https://user-images.githubusercontent.com/21222766/158001272-4dd338a6-01c4-4b01-a88a-a99ab7beabef.png">


Can we use pretarining ?



