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



