# Synthetic Consciousness: Aleksander's Five Axioms, with an Ablation Test

A small, self-contained Java program that implements Igor Aleksander's five axioms of a synthetic mind (presence, imagination, attention, planning and emotion) in a way where **each axiom measurably changes the agent's behaviour**, and where each one can be switched off to prove it.

It is a single file with no dependencies.

## Why this exists

It is easy to write a program with five classes named after the axioms and a loop that calls them in turn. It is much harder to make the axioms *matter*. In many demonstrations, deleting the "attention" or "emotion" module would leave the behaviour exactly the same, which means the module was decoration.

This project takes the opposite approach. Every axiom is wired into the decision the agent makes, and the program includes an **ablation experiment**: switch one axiom off, run the agent again, and see whether performance changes. If it doesn't, the axiom wasn't doing any work.

## What this is, and what it isn't

This program implements the *functional architecture* described by the axioms. It makes **no claim that the agent has experience**, or that anyone is "at home". Satisfying the axioms functionally and having phenomenal consciousness are different questions, and the gap between them is the hard problem.

The honest framing is: *this is a working model of the architecture Aleksander proposed, with evidence that each part contributes to behaviour.* Whether such a system could ever be conscious is left open.

Also note that Aleksander's own systems were built on **weightless (RAM-based) neural networks**. Here the learned state machine is a plain table of states, transitions and values. It captures the idea but not his hardware.

## Quick start

You need Java 11 or later. It was developed and tested on OpenJDK 21.

```bash
# Run the ablation experiment (takes a few seconds)
java SyntheticConsciousness.java

# Watch one trained episode, step by step
java SyntheticConsciousness.java trace

# Watch the same episode with an axiom switched off
java SyntheticConsciousness.java trace --no-attention
java SyntheticConsciousness.java trace --no-imagination
java SyntheticConsciousness.java trace --no-planning
java SyntheticConsciousness.java trace --no-emotion
```

Flags can be combined, for example `trace --no-attention --no-emotion`.

If you prefer to compile first:

```bash
javac SyntheticConsciousness.java
java SyntheticConsciousness
```

## The task

The agent lives in a corridor 15 cells long and has four actions: `LEFT`, `RIGHT`, `EAT` and `REST`.

- **Channel 0** is a food gradient: it tells the agent how far away the food is, and in which direction.
- **Channels 1 to 3** are pure noise, with new random values every step.
- Moving costs a little (-0.02), eating in the wrong place is penalised (-0.1), resting is free, and eating at the food earns +1. The food then respawns somewhere else.
- Each episode is 40 steps, with a random start position and a random food location.

The agent is never told which channel is useful, what the actions do, or where reward comes from. It has to find all of that out.

## How each axiom works

Aleksander numbers the axioms 1 Presence, 2 Imagination, 3 Attention, 4 Planning, 5 Emotion. The code keeps his numbers in its labels but runs them in the order the data flows, because attention decides what the later stages get to represent.

| Axiom | What it does here | What depends on it |
|---|---|---|
| **1 Presence** | The agent senses the world and holds a current sensory state. | Everything else needs a state to work on. |
| **3 Attention** | Tracks how much each channel changes from step to step. Channels that change smoothly are attended to; jumpy channels are ignored. Only attended channels become part of the agent's state. | A compact state lets experience generalise. Without it, noise makes every state look new. |
| **2 Imagination** | Learns a table of *what happens next* for every action in every state, then uses it to predict the consequences of actions without taking them. | The planner needs predicted outcomes to compare. |
| **4 Planning** | Searches three imagined moves ahead and picks the first move of the best imagined sequence. | Lets the agent act on rewards that are several steps away. |
| **5 Emotion** | Learns an affective **value** for each state, and keeps a **mood** (a running sense of how well things are going) that raises the exploration rate when things go badly. | Gives the planner something to evaluate imagined futures against. |

## The ablation experiment

Running `java SyntheticConsciousness.java` trains 20 agents for 300 episodes each, for every configuration, and reports reward per episode. The numbers below come from that run (mean over 20 runs; standard errors were 0.02 to 0.14).

| Configuration | First 50 episodes | Last 50 episodes |
|---|---|---|
| All five axioms | +3.01 | +4.76 |
| No attention | -1.05 | -1.05 |
| No imagination | -0.07 | -0.01 |
| No planning | +1.50 | +4.80 |
| No emotion | +0.54 | +0.33 |
| Scripted walk-to-food (ceiling) | +5.43 | |

Random number generators are seeded, so the results are reproducible.

### Reading the results

- **No attention** is slightly worse than a random walker. The noise channels leak into the agent's state, so it almost never sees the same state twice, never learns, and stores thousands of states where the full agent needs about sixty.
- **No imagination** leaves the agent purely reactive. It can only use what it remembers about immediate rewards, and since moving costs a little and resting is free, it concludes that doing nothing is best.
- **No emotion** leaves the agent with no sense of which imagined states are good. It can still see a couple of moves ahead using remembered rewards, but nothing farther.
- **No planning** is the interesting exception. The agent learns more slowly (+1.50 versus +3.01 early on) but ends up just as good. In this simple corridor, planning speeds up learning but doesn't change where learning ends up. A world with longer horizons or obstacles would be needed to make it matter at the end.

## Watching the agent think

The `trace` mode prints the five axioms at work on each step of a trained agent. For example:

```
step 1   (food is -11 cells away)
  [1 Presence]    sensors = [-0.79 +0.69 +0.49 -0.34]
  [3 Attention]   attending to channels [0]  ->  state "-13|_|_|_|"
  [2 Imagination] LEFT -> -11|_|_|_|   RIGHT -> -14|_|_|_|   EAT -> -13|_|_|_|   REST -> -13|_|_|_|
  [4 Planning]    lookahead 3:  LEFT +0.13   RIGHT +0.07   EAT +0.01   REST +0.11
  [5 Emotion]     mood +0.57   value of this state +0.09   explore-rate 0.09
  => LEFT
```

The agent ignores the three noise channels, predicts where each action would take it, scores the imagined futures, and heads towards the food. Compare this with `trace --no-attention`, where the state is cluttered with noise, the imagination column shows only `?`, and every action scores the same.

## Parameters

These constants sit at the top of `SyntheticConsciousness.java`.

| Constant | Default | Meaning |
|---|---|---|
| `GAMMA` | 0.9 | How much future reward counts relative to immediate reward |
| `ALPHA` | 0.3 | Learning rate for state values |
| `OPTIMISM` | 0.05 | Untried actions look mildly promising, which drives exploration |
| `EPISODE_STEPS` | 40 | Length of each episode |
| `PLAN_DEPTH` | 3 | How many imagined moves the planner looks ahead |

The attention threshold (`STABLE_BELOW`) and warm-up length (`WARMUP`) are inside the `Agent` class.

## Limitations

- **The attention rule is a heuristic.** It prefers channels that change smoothly. A relevant channel that happens to look jumpy would be missed, and a smooth but irrelevant one would fool it.
- **The emotion switch bundles two things:** state values and mood-driven exploration. The experiment can't separate their individual contributions.
- **The axioms interlock.** Imagining futures is useless without some way to value them, so the no-imagination and no-emotion results are not fully independent. That is arguably in the spirit of the axioms, but it means the ablations are not clean single-factor tests.
- **The world is small and fully learnable.** It is designed to make each axiom visibly necessary, not to be a realistic environment.
- **The state machine is tabular,** not Aleksander's weightless neural networks.

## Ideas for extending it

- Add a harder environment (obstacles, a moving predator) so that planning matters at the end of training, not only during it.
- Split the emotion switch into separate "affective value" and "mood" switches.
- Replace the stability heuristic with an attention mechanism that learns relevance from reward.
- Replace the tables with a weightless neural state machine, closer to Aleksander's own architecture.
- Add a new ablation switch: define a field in `Config`, an entry in `Config.without`, and use the flag wherever the faculty acts.

## Background

- Aleksander, I. and Dunmall, B. (2003). *Axioms and Tests for the Presence of Minimal Consciousness in Agents.* Journal of Consciousness Studies, 10(4-5), 7-18.
- Aleksander, I. (1996). *Impossible Minds: My Neurons, My Consciousness.* Imperial College Press.
- Aleksander, I. (2005). *The World in My Mind, My Mind in the World: Key Mechanisms of Consciousness in People, Animals and Machines.* Imprint Academic.

## License

Add your preferred license here (MIT is a common choice for small projects like this).
