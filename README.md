# A Deep-Learning Proposition for Computational Missile Guidance

The project explores using **Deep Reinforcement Learning** to develop a computational guidance law for intercepting highly maneuvering targets with lateral acceleration constraints. By formulating the problem within an RL framework, we aim to generate optimal guidance commands. The study highlights the potential of deep RL in enhancing guidance algorithms and addressing the complexities of intercepting maneuvering targets. 

The following assumptions are made while formulating the guidance problem: <br>
-The linear speeds of the missile and the target are constant and known.  <br>
-Missile autopilot dynamics is a first-order time lag system.  <br>
-The missile and target engagement occurs in a 2-D vertical plane.  <br>

We establish a framework in RL by incorporating the engagement kinematics as the environment and the guidance command as the agent action. The reward function design ensures the missile’s rapid and stable interception. To achieve this, we employ the *state-of-the-art* **Deep Deterministic Policy Gradient** method to learn a policy that directly maps observed states to guidance commands.
