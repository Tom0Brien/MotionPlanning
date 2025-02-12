classdef PLANNER < handle
    properties
        model          % f(u,dt,x)
        N              % Control horizon
        I              % Number of samples
        dt             % Time step
        lambda         % Temperature parameter
        sigma_epsilon  % Control noise covariance
        n_states       % Number of states 
        n_actions      % Number of actions (inputs, controls)
        U              % Mean control sequence
        stage_cost     % User defined stage cost function
        terminal_cost  % User defined terminal cost function
        rollouts       % Set of rollouts in control update
        weights        % Weights after importance sampling
        nu
        R
        deltaPosMax    % Maximum allowed position change per step
        deltaOrientMax % Maximum allowed orientation change per step
    end
    
    methods
        function self = PLANNER(model, N, I, dt, lambda, sigma_epsilon, nu, R, n_states, n_actions, stage_cost, terminal_cost, deltaPosMax, deltaOrientMax)
            self.model = model;
            self.N = N;
            self.I = I;
            self.dt = dt;
            self.lambda = lambda;
            self.sigma_epsilon = sigma_epsilon;
            self.n_states = n_states;
            self.n_actions = n_actions;
            self.U = zeros(n_actions, N);
            self.stage_cost = stage_cost;
            self.terminal_cost = terminal_cost;
            self.nu = nu;
            self.R = R;
            self.deltaPosMax = deltaPosMax;
            self.deltaOrientMax = deltaOrientMax;
        end

        function [] = set_action(self, U)
            self.U = U;
        end

        function [U_opt] = get_action(self, x0)
            % GET_ACTION_FMINCON Solve a direct optimization problem using fmincon
            % to find the control sequence over the horizon.
            %
            %   x0    = current state (n_states x 1)
            %   U_opt = optimal control sequence (n_actions x N)
            
            % Flatten the previous solution self.U to use as an initial guess
            U0 = self.U(:);  % (n_actions*N x 1)
            
            % Define the cost function handle
            costFun = @(Uvar) trajectory_cost(Uvar, x0, self.model, ...
                self.stage_cost, self.terminal_cost, self.N, self.dt, ...
                self.n_actions);

            % ----------------- Updated bounds for position and orientation ----------------
            % For each control step, the first 3 entries (position) have bounds ±deltaPosMax
            % and the last 3 entries (orientation) have bounds ±deltaOrientMax.
            lb_step = [-self.deltaPosMax; -self.deltaPosMax; -self.deltaPosMax; ...
                       -self.deltaOrientMax; -self.deltaOrientMax; -self.deltaOrientMax];
            ub_step = [ self.deltaPosMax;  self.deltaPosMax;  self.deltaPosMax; ...
                        self.deltaOrientMax;  self.deltaOrientMax;  self.deltaOrientMax];
            lb = repmat(lb_step, self.N, 1);
            ub = repmat(ub_step, self.N, 1);
            % ----------------------------------------------------------------------------------

            % If needed, linear or nonlinear constraints go here
            A = []; b = [];
            Aeq = []; beq = [];
            nonlcon = [];  % or define a function handle for state constraints
            
            % Solve the NLP
            options = optimoptions('fmincon', 'Display', 'off'); %,'MaxIterations',100,'MaxFunctionEvaluations',100);
            [Usol, fval, exitflag] = fmincon(costFun, U0, A, b, Aeq, beq, lb, ub, nonlcon, options);
            
            % Reshape solution into (n_actions x N)
            U_mat = reshape(Usol, [self.n_actions, self.N]);
            
            % Receding-horizon approach: we only apply the first control in U_mat,
            % then shift for warm-start in next iteration
            U_opt = U_mat;
            
            % Warm-start (shift) for next call:
            self.U = [U_mat(:,2:end), zeros(self.n_actions,1)];
        end
    end
end

function J = trajectory_cost(Uvar, x0, model, stage_cost, terminal_cost, N, dt, n_actions)
    % Reshape the decision variable into matrix form
    U_mat = reshape(Uvar, [n_actions, N]);
    
    % Forward simulate the system
    x = x0;
    J = 0;  % accumulate cost
    for k = 1:N
        % Current control
        u_k = U_mat(:,k);
        
        % Step the system
        x = model.step(u_k, dt, x);
        
        % Add stage cost
        J = J + stage_cost(x, u_k);
    end
    
    % Add terminal cost based on final state (and possibly final control)
    J = J + terminal_cost(x, U_mat(:,end));
end