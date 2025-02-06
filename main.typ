#import "@preview/touying:0.5.5": *
#import themes.dewdrop: *
#import "@preview/numbly:0.1.0": numbly
#set heading(numbering: numbly("{1}.", default: "1.1"))


#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  navigation: "mini-slides",
  mini-slides: (
    height: 4em, 
    x: 3.3em,
    display-section: false,
    display-subsection: true,
    short-heading: false
  ),
  // You can add or override more theme settings here, e.g.:
  config-common(slide-level: 2),
  config-colors(
    neutral-darkest: rgb("#000000"),
    neutral-dark: rgb("#202020"),
    neutral-light: rgb("#f3f3f3"),
    neutral-lightest: rgb("#ffffff"),
    primary: rgb("#00573F"),   
    secondary: rgb("#D4AF37")
  ),
  config-info(
    title: [Neural Episodic Control (NEC)],
    subtitle: [Pritzel et al. (2017) | https://arxiv.org/abs/1703.01988],
    author: [David Bayha, Dhruv Dhamani, Geet Saurabh Kunumala, Yingjie Ma],
    date: datetime.today(),
    institution: [College of Computing and Informatics \ University of North Carolina at Charlotte],
  ),
  footer: [ITCS 6050 | 8050 – Deep Reinforcement Learning | Spring 2025],
  footer-right: [#image("uncc-logo.png")],
  da: datetime(day: 6, month: 2, year: 2025),
)

#title-slide()

//-------------------------------------
// Slide 2
//-------------------------------------
= Introduction

== Can AI learn as fast as Humans? 

#grid(
  columns: (4fr, 2fr),
  rows: (auto, 60pt),
  gutter: 3pt,
  [- *Statistic*: In the Atari 2600 set of environments, deep Q-networks required *more than 200 hours* of gameplay in order to achieve scores similar to a human who only played 2 hours. (Bellemare et al., 2013)

  - Imagine a human only needing one example shown to learn something, but an AI needs to see that same example millions of times.

DL and Episodic vs. Semantic Memory (1:20-2:40): #text(link("https://www.youtube.com/watch?v=9T9DsH4K0ik&t=80s")[Watch on YouTube], fill: green)
],
  [ #set align(right)
    #image("stock-image.png", height: 51%)]
)


//-------------------------------------
// Slide 3
//-------------------------------------
== Problem

- *What is the Problem?*  

Deep Reinforcement Learning (DRL) can exceed human performance, but it requires significantly more interactions to learn, making the process *slow* and *inefficient*.

- *What is the Solution?*  
Neural Episodic Control (NEC) is a technique that allows a learning agent to learn faster by remembering past experiences.  
  - NEC stores and *recalls past successful actions* instead of learning purely through trial and error.  
  - Inspired by the hypothesized role of the hippocampus in decision-making.


//-------------------------------------
// Slide 4
//-------------------------------------
== Problem Statement

- *What is the purpose of the paper?*  
Provides an answer to the question:\
_Why are Deep Reinforcement-Learning agents so slow at learning?_

- Seeks to address three major challenges:
  + *Stochastic Gradient Descent (SGD)* optimization requires using small learning rates – large learning rates cause catastrophic interference (forgetting).
  + Environments with a *sparse reward signal* make it difficult for a RL agent to learn its environment – there may be very few instances with a non-zero reward.  
  + Many DRL agents using value-based RL methods (ie. Q-learning) learn one step at a time, resulting in *slow reward signal propagation* – _agent may take hundreds of steps before retrieving useful information_

//-------------------------------------
// Slide 5
//-------------------------------------
== Motivation

- In order for DRL techniques to be applicable to real-world problems it is essential that faster learning occurs
- *Neural Episodic Control* (NEC) dramatically improves the _efficiency_ of RL agents by _storing/recalling successful past experiences_ and reducing trial-and-error learning
- Some potential real-world applications include:
  - *Robotics*: Robots require fast learning of their environment
  - *Healthcare*: optimize health care decisions and more efficiently personalize healthcare plans
  - *Autonomous vehicles* commonly in high-speed environments

//-------------------------
// Slide 6
//-------------------------
// = Background

// == DQN and Reinforcement Learning

// - Reinforcement Learning is a framework for learning optimal actions through interactions with environment to maximize reward.  
// - DQN uses Q-learning to learn value function $Q(s_t, a_t)$  
// - $Q(s_t, a_t)$ takes 2D pixel representation of state st and outputs vector containing value of each action at that state. 
// - Upon observation, DQN stores $(s_t, a_t, r_t, s_t + 1)$ tuple in replay buffer, which is used for training.

// //-------------------------
// // Slide 7
// //-------------------------
// == Improvements on DQN

// - Double DQN decouples action selection and action evaluation steps to reduce overestimation bias.  
// - Prioritized Replay further improves on Double DQN by optimizing replay strategy.  
// - Many papers have suggested that switching to on-policy learning allows agent to learn faster in Atari environments  
// - AC3 works on policy gradient, which learns a policy and its associated value function.

// //-------------------------
// // Slide 8
// //-------------------------
// == Neural Episodic Control.

// - NEC rapidly latches onto successful strategies as soon as they are experienced, instead of waiting many steps.

// - The Agent has 3 components:
//   + Convoluted Neural Network -  Processes pixel images.
//   + Set of memory modes - One per action. 
//   + Final Network - Convert action memories into $Q(s, a)$ values.

// - For each action, NEC has a memory module with key-value pairs called differentiable neural dictionary (DND)

// //-------------------------
// // Slide 9
// //-------------------------
// == Differentiable Neural Dictionary

// - DND has 2 operations, lookup and write  
// - The output of lookup is a weighted sum of values in memory, whose weights are given by normalized kernels between lookup key and corresponding key in memory.  
// - After DND is queried, new key-value pair is written into memory. Writes are append-only.

= Background

== Deep Reinforcement Learning Overview

- *Reinforcement Learning*: learning optimal actions through environment interactions to maximize cumulative reward.
- *Deep RL*: neural networks approximate value or policy functions.
- *Data Inefficiency*: these methods often require millions of frames before converging to human-level performance.

==  Deep Q-Network Basics

- *Deep Q-Network:*
  - Learns a value function $Q(s,a)$ via Q-learning.
  - Takes raw pixels; outputs action values.
  - Uses experience replay (stores  $(s,a,r,s')$ tuples) and target networks for stable learning.

- *Limitations of DQN*
  - Very large data requirements (thousands of hours on Atari).
  - Slow credit assignment (one-step Q-learning).

== Improvements on DQN

- *Double DQN:* decouples action selection from action evaluation → reduces Q-value overestimation.
- *Prioritized Experience Replay*: replays important transitions more frequently → speeds up learning.
- *On-policy Methods* (e.g., A3C): can help with credit assignment, but also have data demands.
- *Other Techniques*: $"Retrace"(λ)$, $Q*(λ)$, hierarchical RL, etc. → still not as fast as humans in practice.

== Motivation for Memory-Based Approaches

- *Memory in RL:*
  - Classic tabular RL can update values quickly, but doesn’t scale well to large state spaces.
  - Neural networks scale to high-dimensional inputs, but slow updates hamper data efficiency.
- *Episodic/MFEC:*
  - Model-Free Episodic Control (Blundell et al., 2016) shows a memory-based method can drastically speed learning by storing experienced states and their returns.
- *Key Idea:* A more flexible episodic memory could combine the best of fast local updates (like tabular RL) with generalization from deep learning.

//-------------------------
// Slide 10
//-------------------------

= Methods
== Overview of Neural Episodic Control

- _Three Main Components:_
  + *CNN Embedding Network* for state representation  
  + *Memory Module (DND)*: one per action  
  + *Final Network* to combine memory outputs into $Q(s,a)$

- _Key Idea:_ Store (key, value) pairs in a large external memory  
  - *Keys* = slow-changing embeddings from CNN  
  - *Values* = fast-updated action-value estimates  

- _Motivation:_ “Episodic memory” allows rapid assimilation of new experiences

== Differentiable Neural Dictionary (DND)

- _Differentiable Neural Dictionary_ = Key-Value Store
  - *Lookup*: find nearest keys to current embedding $h$
  - *Weighted Sum*: output value is a kernel-weighted average of stored values

- *Memory Growth*: append-only writes; update existing entries if the key already exists

- *Efficient Retrieval*: approximate nearest neighbor search (e.g., *kd-trees*) allows large-scale memory


== N-step Updates & Training Loop

- *N-step* Q-learning for *faster reward propagation*:

$ Q^(N)(s_t, a_t) = sum_(j=0)^(N-1) gamma^j r_(t+j) + gamma^N max_(a') Q(s_(t+N), a') $

- *Memory Update:*

$ Q_i <- Q_i + alpha(Q^(N) - Q_i) $

- *Replay Buffer*: small buffer to train the CNN embedding; slow gradient updates to refine representation.

== Experimental Setup

- 57 *Atari 2600* games (Arcade Learning Environment)
- Training from *1M to 40M* frams of gameplay
$ "Human-Normalized Score (HNS)" = ("score"_"agent" - "score"_"random")/("score"_"human" - "score"_"random") $
- Recorded performance at specific checkpoints: *1M, 2M, 4M, 10M, 20M*, and *40M frames*
- Compared with: *DQN, Double DQN, Prioritized Replay, A3C, MFEC*
- NEC uses same CNN architecture as DQN for fair comparison

= Results

== Median Human-Normalized Score

- NEC uses same CNN architecture as DQN for fair comparison

#align(center)[#image("table-2.png", height: 78%)]

#align(center)[#image("figure-4.png")]

#align(center)[#image("figure-8.png")]

#grid(columns: 2, [
  #image("fig-pac-man.png", height: 87%)
  ],
  [
  #image("fig-alien.png", height: 87%)
  ]
)

== Why does NEC learn faster?

- *Memory-augmented reinforcement learning:*
  - NEC stores and retrieves past experiences efficiently using a Differentiable Neural Dictionary (DND).
- *N-step Q-learning for faster reward propagation:*
  - Unlike traditional Q-learning, NEC combines Monte Carlo estimates with off-policy learning, leading to quicker adaptation.
- *Avoiding slow updates in deep networks:*
  - DQN and A3C rely on slow gradient-based learning, while NEC updates stored Q-values instantly.

== Takeaways

+ *NEC is ideal for fast adaptation*
  - Achieves human-level performance 10x faster than DQN and A3C.
  - Excels in low-data regimes where interactions with the environment are limited.
+ *NEC is memory-efficient*
  - Uses non-parametric memory retrieval instead of storing everything in a neural network.
  - This allows rapid learning without extensive retraining.
+ *NEC’s performance advantage fades in long-term training*
  - While NEC is best in early learning, Prioritized Replay DQN outperforms it after 40M frames.
  - Future work should combine NEC’s fast learning with DQN’s stability for long-term gains.

== Conclusion

- Neural Episodic Control (NEC) offers a breakthrough in reinforcement learning efficiency.
- It learns significantly faster than existing methods but requires enhancements for long-term stability.
- The study highlights the potential of memory-based approaches in AI decision-making.

= Discussion

== Clarification

== Analysis

== Application

== Reflection