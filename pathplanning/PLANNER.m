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
    end
    
    methods
        function self = PLANNER(model, N, I, dt, lambda, sigma_epsilon, nu, R, n_states,n_actions,stage_cost, terminal_cost)
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
        end

        function [] = set_action(self, U)
            self.U = U;
        end
        
        function [U_opt] = get_action(self, x0)
            S = zeros(1, self.I);
            epsilons = zeros(self.n_actions, self.N, self.I);
            self.rollouts = zeros(self.n_states, self.N, self.I);
            
            % Rollouts
            for i = 1:self.I
                x = x0;
                epsilons(:, :, i) = mvnrnd(zeros(1,self.n_actions), self.sigma_epsilon,self.N).';
                for k = 1:self.N
                    x = self.model.step(self.U(:, k) + epsilons(:, k, i), self.dt, x);
                    self.rollouts(:, k, i) = x;
                    q_cost = self.stage_cost(x,self.U(:, k) + epsilons(:, k, i));
                    S(i) = S(i) + q_cost + self.lambda * self.U(:, k)' * (self.sigma_epsilon \ epsilons(:, k, i));
                    % S(i) = S(i) + q_cost + (1-1/self.nu)/2 * epsilons(:, k, i)'*self.R*epsilons(:, k, i) + (self.U(:, k) + epsilons(:, k, i))'*self.R*epsilons(:, k, i) + 1/2*(self.U(:, k) + epsilons(:, k, i))'*self.R*(self.U(:, k) + epsilons(:, k, i));
                end
                % Add terminal cost
                S(i) = S(i) + self.terminal_cost(x,self.U(:, k) + epsilons(:, self.N, i));
            end
            
            % Importance sampling
            beta = min(S);
            self.weights = exp(-1 / self.lambda * (S - beta));
            self.weights = self.weights / sum(self.weights);
            U_opt = self.U;
            for k = 1:self.N
                du = 0;
                for i = 1:self.I
                    du = du + self.weights(i) * epsilons(:, k, i);
                end
                U_opt(:,k) = U_opt(:,k) + du;
            end
            self.U(:,1:end-1) = U_opt(:,2:end);
        end

        function [U_opt] = get_action_fmincon(self, x0)
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
            
            % (Optional) define bounds or constraints on U
            % Example: let each control be within [-1, 1] in every dimension
            lb = -1 * ones(self.n_actions*self.N, 1);
            ub =  1 * ones(self.n_actions*self.N, 1);
            
            % If needed, linear or nonlinear constraints go here
            A = []; b = [];
            Aeq = []; beq = [];
            nonlcon = [];  % or define a function handle for state constraints
            
            % Solve the NLP
            [Usol, fval, exitflag] = fmincon(costFun, U0, A, b, Aeq, beq, lb, ub, nonlcon);
            
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