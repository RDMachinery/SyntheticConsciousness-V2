import java.util.*;

/**
 * Igor Aleksander's five axioms of a synthetic mind, rebuilt so that each axiom
 * is causally load-bearing and can be switched off for an ablation experiment.
 *
 * Usage:
 *   java SyntheticConsciousness                     run the ablation experiment
 *   java SyntheticConsciousness trace               watch one trained episode, step by step
 *   java SyntheticConsciousness trace --no-attention --no-imagination
 *                                                   (also --no-planning, --no-emotion)
 *
 * What the agent has to do: it lives in a corridor, senses a gradient that points
 * to food (channel 0) amid three channels of pure noise (channels 1-3), and earns
 * reward by walking to the food and eating. Food respawns after each meal.
 *
 * Aleksander's numbering is: 1 Presence, 2 Imagination, 3 Attention, 4 Planning,
 * 5 Emotion. The code runs them in the order the data flows (Presence, Attention,
 * Imagination, Planning, Emotion), because attention decides what the later
 * stages get to represent. The labels below keep his numbers.
 *
 * Honest caveats: Aleksander's own systems used weightless (RAM-based) neural
 * networks. Here the learned state machine is a table of states, transitions and
 * values, which captures the idea but not his hardware. The axioms also interlock:
 * imagining futures is useless without some way to value them, which is itself
 * in the spirit of his argument.
 */
public class SyntheticConsciousness {

    static final double GAMMA = 0.9;        // how much the future counts
    static final double ALPHA = 0.3;        // learning rate for affective value
    static final double OPTIMISM = 0.05;    // untried actions look mildly promising
    static final int EPISODE_STEPS = 40;
    static final int PLAN_DEPTH = 3;

    enum Action { LEFT, RIGHT, EAT, REST }

    // ------------------------------------------------------------------
    //  The ablation switches
    // ------------------------------------------------------------------
    static final class Config {
        boolean attention = true, imagination = true, planning = true, emotion = true;

        Config without(String axiom) {
            Config c = new Config();
            c.attention = attention; c.imagination = imagination;
            c.planning = planning;   c.emotion = emotion;
            switch (axiom) {
                case "attention"   -> c.attention = false;
                case "imagination" -> c.imagination = false;
                case "planning"    -> c.planning = false;
                case "emotion"     -> c.emotion = false;
                default -> throw new IllegalArgumentException(axiom);
            }
            return c;
        }
    }

    // ------------------------------------------------------------------
    //  The world
    // ------------------------------------------------------------------
    static final class Corridor {
        static final int LENGTH = 15;
        static final int CHANNELS = 4;       // 0 = food gradient, 1..3 = noise

        private final Random rng;
        private int pos, food;
        private boolean ate;

        Corridor(Random rng) { this.rng = rng; }

        void reset() {
            pos = rng.nextInt(LENGTH);
            food = rng.nextInt(LENGTH);
            ate = false;
        }

        double[] sense() {
            double[] s = new double[CHANNELS];
            s[0] = (food - pos) / (double) (LENGTH - 1);
            for (int i = 1; i < CHANNELS; i++) s[i] = rng.nextDouble() * 2 - 1;
            return s;
        }

        double step(Action a) {
            ate = false;
            if (a == Action.LEFT)  { pos = Math.max(0, pos - 1);          return -0.02; }
            if (a == Action.RIGHT) { pos = Math.min(LENGTH - 1, pos + 1); return -0.02; }
            if (a == Action.EAT) {
                if (pos == food) {
                    ate = true;
                    do { food = rng.nextInt(LENGTH); } while (food == pos);
                    return 1.0;
                }
                return -0.1;
            }
            return 0.0;                                                    // REST
        }

        boolean ate() { return ate; }
        int distanceToFood() { return food - pos; }
    }

    // ------------------------------------------------------------------
    //  What the agent remembers about each state it has been in
    // ------------------------------------------------------------------
    static final class StateInfo {
        final double[] rewardSum = new double[Action.values().length];
        final int[] rewardCount = new int[Action.values().length];
        final Map<String, Integer>[] next;     // learned transitions (Imagination)
        double value = 0;                      // affective value (Emotion)

        @SuppressWarnings("unchecked")
        StateInfo() { next = new Map[Action.values().length]; }

        Map<String, Integer> transitions(int a) {
            if (next[a] == null) next[a] = new HashMap<>();
            return next[a];
        }
    }

    // ------------------------------------------------------------------
    //  The agent
    // ------------------------------------------------------------------
    static final class Agent {
        private static final int WARMUP = 30;           // observations before attention judges
        private static final double STABLE_BELOW = 0.3; // mean change under this = worth attending to

        private final Config cfg;
        private final Random rng;
        private final Map<String, StateInfo> states = new HashMap<>();

        // attention bookkeeping
        private final double[] meanChange = new double[Corridor.CHANNELS];
        private final boolean[] attended = new boolean[Corridor.CHANNELS];
        private int changeSamples = 0;
        private double[] prevRaw = null;

        // presence
        private double[] curRaw;
        private String prevKey = null, curKey = null;

        // emotion
        private double avgReward = 0, mood = 0;

        Agent(Config cfg, Random rng) { this.cfg = cfg; this.rng = rng; }

        void newEpisode() { prevRaw = null; prevKey = null; curKey = null; }

        double mood() { return mood; }
        int knownStates() { return states.size(); }

        // ---- AXIOMS 1 and 3: PRESENCE and ATTENTION -------------------
        // Sense the world, then decide which parts of it become the agent's state.
        void perceive(double[] raw) {
            curRaw = raw.clone();                                    // Axiom 1: Presence

            if (prevRaw != null) {                                   // Axiom 3: Attention
                changeSamples++;
                double rate = Math.max(1.0 / changeSamples, 0.02);
                for (int i = 0; i < raw.length; i++)
                    meanChange[i] += rate * (Math.abs(raw[i] - prevRaw[i]) - meanChange[i]);
            }
            prevRaw = raw.clone();

            StringBuilder key = new StringBuilder();
            for (int i = 0; i < raw.length; i++) {
                attended[i] = !cfg.attention || changeSamples < WARMUP || meanChange[i] < STABLE_BELOW;
                if (attended[i]) key.append(Math.round(raw[i] * 16)); else key.append('_');
                key.append('|');
            }
            prevKey = curKey;
            curKey = key.toString();
        }

        // ---- AXIOMS 2, 4 and 5: IMAGINATION, PLANNING, EMOTION --------
        Action decide(boolean verbose) {
            Action[] acts = Action.values();
            int depth = cfg.planning ? PLAN_DEPTH : 1;
            double[] scores = new double[acts.length];
            for (int a = 0; a < acts.length; a++) scores[a] = score(curKey, a, depth);

            double eps = cfg.emotion ? 0.05 + 0.20 * (1 - mood) / 2 : 0.15;
            boolean explore = rng.nextDouble() < eps;
            int choice;
            if (explore) {
                choice = rng.nextInt(acts.length);
            } else {
                double best = Double.NEGATIVE_INFINITY;
                for (double s : scores) best = Math.max(best, s);
                List<Integer> ties = new ArrayList<>();
                for (int a = 0; a < acts.length; a++) if (scores[a] > best - 1e-9) ties.add(a);
                choice = ties.get(rng.nextInt(ties.size()));
            }

            if (verbose) {
                System.out.printf("  [1 Presence]    sensors = %s%n", fmt(curRaw));
                System.out.printf("  [3 Attention]   attending to channels %s  ->  state \"%s\"%n",
                        attendedList(), curKey);
                StringBuilder im = new StringBuilder();
                StateInfo here = states.get(curKey);
                for (int a = 0; a < acts.length; a++) {
                    String n = cfg.imagination ? imagine(here, a) : null;
                    im.append(acts[a]).append(" -> ").append(n == null ? "?" : n).append("   ");
                }
                System.out.printf("  [2 Imagination] %s%n", cfg.imagination ? im.toString().trim() : "(switched off)");
                StringBuilder sc = new StringBuilder();
                for (int a = 0; a < acts.length; a++) sc.append(String.format("%s %+.2f   ", acts[a], scores[a]));
                System.out.printf("  [4 Planning]    lookahead %d:  %s%n", depth, sc.toString().trim());
                System.out.printf("  [5 Emotion]     mood %+.2f   value of this state %+.2f   explore-rate %.2f%n",
                        cfg.emotion ? mood : 0.0, valueOf(curKey), eps);
                System.out.printf("  => %s%s%n", acts[choice], explore ? "  (exploratory)" : "");
            }
            return acts[choice];
        }

        // Value of taking action a in a state, looking `depth` moves ahead in imagination.
        private double score(String key, int a, int depth) {
            StateInfo s = states.get(key);
            double r = (s == null || s.rewardCount[a] == 0) ? OPTIMISM : s.rewardSum[a] / s.rewardCount[a];
            if (!cfg.imagination) return r;                          // no foresight: react to memory only

            String next = imagine(s, a);                             // Axiom 2: Imagination
            if (next == null) next = key;                            // unknown consequences: assume no change
            double future;
            if (depth <= 1) {
                future = valueOf(next);                              // Axiom 5: affective evaluation
            } else {
                future = Double.NEGATIVE_INFINITY;                   // Axiom 4: Planning
                for (int a2 = 0; a2 < Action.values().length; a2++)
                    future = Math.max(future, score(next, a2, depth - 1));
            }
            return r + GAMMA * future;
        }

        private String imagine(StateInfo s, int a) {
            if (s == null || s.next[a] == null || s.next[a].isEmpty()) return null;
            String best = null; int bestCount = -1;
            for (Map.Entry<String, Integer> e : s.next[a].entrySet())
                if (e.getValue() > bestCount) { best = e.getKey(); bestCount = e.getValue(); }
            return best;
        }

        private double valueOf(String key) {
            if (!cfg.emotion) return 0;
            StateInfo s = states.get(key);
            return s == null ? 0 : s.value;
        }

        // ---- Learning from what actually happened ----------------------
        void learn(Action action, double reward, boolean ate) {
            if (cfg.emotion) {                                       // Axiom 5: mood follows reward
                avgReward += 0.05 * (reward - avgReward);
                mood = Math.tanh(8 * avgReward);
            }
            if (prevKey == null) return;
            int a = action.ordinal();
            StateInfo s = states.computeIfAbsent(prevKey, k -> new StateInfo());
            s.rewardSum[a] += reward;
            s.rewardCount[a]++;
            if (cfg.imagination) s.transitions(a).merge(curKey, 1, Integer::sum);
            if (cfg.emotion) {
                double target = ate ? reward : reward + GAMMA * valueOf(curKey);
                s.value += ALPHA * (target - s.value);
            }
        }

        private String attendedList() {
            List<Integer> on = new ArrayList<>();
            for (int i = 0; i < attended.length; i++) if (attended[i]) on.add(i);
            return on.toString();
        }

        private static String fmt(double[] v) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < v.length; i++) sb.append(i > 0 ? " " : "").append(String.format("%+.2f", v[i]));
            return sb.append("]").toString();
        }
    }

    // ------------------------------------------------------------------
    //  Running things
    // ------------------------------------------------------------------
    static final class Session {
        final Corridor env;
        final Agent agent;

        Session(Config cfg, long seed) {
            env = new Corridor(new Random(seed));
            agent = new Agent(cfg, new Random(seed ^ 0x9E3779B97F4A7C15L));
        }

        double episode(boolean verbose, int steps) {
            env.reset();
            agent.newEpisode();
            agent.perceive(env.sense());
            double total = 0;
            for (int t = 0; t < steps; t++) {
                if (verbose) System.out.printf("%nstep %d   (food is %+d cells away)%n", t + 1, env.distanceToFood());
                Action a = agent.decide(verbose);
                double r = env.step(a);
                total += r;
                agent.perceive(env.sense());
                agent.learn(a, r, env.ate());
                if (verbose) System.out.printf("  outcome: reward %+.2f%s%n", r, env.ate() ? "  (ate!)" : "");
            }
            return total;
        }
    }

    // A scripted agent that simply walks to the food: the ceiling for this task.
    static double oracleEpisode(Random rng) {
        Corridor env = new Corridor(rng);
        env.reset();
        double total = 0;
        for (int t = 0; t < EPISODE_STEPS; t++) {
            int d = env.distanceToFood();
            total += env.step(d > 0 ? Action.RIGHT : d < 0 ? Action.LEFT : Action.EAT);
        }
        return total;
    }

    static double mean(double[] x, int from, int to) {
        double s = 0; for (int i = from; i < to; i++) s += x[i]; return s / (to - from);
    }

    static double stdErr(double[] x) {
        double m = mean(x, 0, x.length), v = 0;
        for (double d : x) v += (d - m) * (d - m);
        return Math.sqrt(v / (x.length - 1)) / Math.sqrt(x.length);
    }

    static void ablationExperiment() {
        final int seeds = 20, episodes = 300, window = 50;
        Config all = new Config();
        String[] names = { "all five axioms", "no attention", "no imagination", "no planning", "no emotion" };
        Config[] configs = {
            all, all.without("attention"), all.without("imagination"),
            all.without("planning"), all.without("emotion")
        };

        System.out.printf("Reward per %d-step episode, mean +/- standard error over %d runs%n", EPISODE_STEPS, seeds);
        System.out.printf("%-18s %-18s %-18s %s%n", "configuration", "first " + window + " episodes",
                "last " + window + " episodes", "states stored");
        for (int c = 0; c < configs.length; c++) {
            double[] early = new double[seeds], late = new double[seeds], stored = new double[seeds];
            for (int s = 0; s < seeds; s++) {
                Session session = new Session(configs[c], 1000 + s);
                double[] totals = new double[episodes];
                for (int e = 0; e < episodes; e++) totals[e] = session.episode(false, EPISODE_STEPS);
                early[s] = mean(totals, 0, window);
                late[s] = mean(totals, episodes - window, episodes);
                stored[s] = session.agent.knownStates();
            }
            System.out.printf("%-18s %+6.2f +/- %-8.2f %+6.2f +/- %-8.2f %.0f%n", names[c],
                    mean(early, 0, seeds), stdErr(early), mean(late, 0, seeds), stdErr(late),
                    mean(stored, 0, seeds));
        }
        double[] o = new double[2000];
        Random rng = new Random(7);
        for (int i = 0; i < o.length; i++) o[i] = oracleEpisode(rng);
        System.out.printf("%-18s %+6.2f   (scripted walk-to-food: the ceiling)%n", "reference", mean(o, 0, o.length));
    }

    static void trace(Config cfg) {
        Session session = new Session(cfg, 42);
        for (int e = 0; e < 200; e++) session.episode(false, EPISODE_STEPS);
        System.out.println("Agent trained for 200 episodes. Watching one more (first 15 steps).");
        session.episode(true, 15);
    }

    public static void main(String[] args) {
        if (args.length > 0 && args[0].equals("trace")) {
            Config cfg = new Config();
            for (String a : args) {
                switch (a) {
                    case "--no-attention"   -> cfg.attention = false;
                    case "--no-imagination" -> cfg.imagination = false;
                    case "--no-planning"    -> cfg.planning = false;
                    case "--no-emotion"     -> cfg.emotion = false;
                    default -> { }
                }
            }
            trace(cfg);
        } else {
            ablationExperiment();
        }
    }
}
