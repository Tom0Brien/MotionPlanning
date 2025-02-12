# Optimization Problem for 3D Planning with PBVS

## Problem Formulation

Given a dynamic system with state \( x \in \mathbb{R}^n \) and control input \( u \in \mathbb{R}^m \), we seek an optimal control sequence over a horizon \( N \) that minimizes a cost function while ensuring the system follows a desired trajectory.

### **Objective Function**

Minimize the total cost over the planning horizon:

\[
J(U) = \sum\_{k=0}^{N-1} \ell(x_k, u_k) + \Phi(x_N, u_N)
\]

where:

- \( \ell(x_k, u_k) \) is the **stage cost** at each time step,
- \( \Phi(x_N, u_N) \) is the **terminal cost** at the final state.

### **State Dynamics**

The system evolves according to the discrete-time model:

\[
x\_{k+1} = f(x_k, u_k, \Delta t)
\]

where \( f \) is the system's motion model and \( \Delta t \) is the time step.

### **Constraints**

The control input is subject to constraints:

\[
u*{\min} \leq u_k \leq u*{\max}, \quad \forall k = 0, \dots, N-1
\]

where \( u*{\min} \) and \( u*{\max} \) define the allowed range for each control input.

### **Optimization Formulation**

\[
\begin{aligned}
\min*{U} \quad & J(U) = \sum*{k=0}^{N-1} \ell(x*k, u_k) + \Phi(x_N, u_N) \\
\text{subject to} \quad & x*{k+1} = f(x*k, u_k, \Delta t), \quad k = 0, \dots, N-1 \\
& u*{\min} \leq u*k \leq u*{\max}, \quad \forall k
\end{aligned}
\]

where:

- \( U = \{u*0, u_1, \dots, u*{N-1}\} \) is the control sequence to be optimized,
- \( x_k \) is the system state at time step \( k \),
- \( u_k \) is the control input at time step \( k \).

### **Numerical Optimization**

The problem is solved using `fmincon`, a nonlinear optimization solver, which minimizes the objective function subject to system dynamics and constraints.

### **Implementation in MATLAB**

The optimization is performed using MATLAB's `fmincon`, where:

- **Decision variable**: \( U \) is a vector of control inputs flattened over the horizon.
- **Cost function**: Evaluates trajectory cost based on state evolution.
- **Constraints**: Box constraints for control inputs and (optionally) nonlinear constraints on state trajectories.
