function example_3D_planning_with_pointcloud()
    clear; clc; close all;
    
    % ------------------------ Define Color Scheme ------------------------
    colors_struct = struct(...
        'PaleGrey1', '#EAF0F9', ...
        'LightBlueGrey', '#C7D6ED', ...
        'PaleGrey2', '#EAF9FE', ...
        'RobinsEggBlue', '#8DD6EF', ...
        'White', '#FFFFFF', ...
        'DuckEggBlue', '#B5E5D6', ...
        'OffWhite', '#F3F8E9', ...
        'Eggshell', '#CFE9C2', ...
        'VeryLightPink1', '#FFF6EE', ...
        'LightPeach', '#FFDDC5', ...
        'VeryLightPink2', '#FFECED', ...
        'PaleRose', '#FFB7B7', ...
        'LightPink', '#FFDBDF', ...
        'RosePink', '#F8A9A9', ...
        'Black', '#000000'...
    );

    % --------------------- 1) Load and Downsample the Point Cloud -------
    ptCloud= pcread('vine_simple.pcd');
    collisionRadius = 0.01;  

    % --------------------- 2) 3D Environment & Problem Setup ------------
    goals = [
        0.23958  0.27   0.38      0.43975             ;   % x-coords
        0.7      0.7   0.74      0.7                ;   % y-coords
        0.6098   0.6386 0.66107   0.65                % z-coords
    ];
    Delta_max = 0.05;

    % --------------------- PLANNER Hyperparameters --------------------------
    dt        = 0.5;            
    N         = 5;              
    I         = 250;           
    lambda    = 0.5;            
    sigma_eps = (1e-2^2)*eye(3);
    nu        = 1.0;            
    R         = 0.1*eye(3);     

    % --------------------- Model Definition ------------------------------
    model = struct();
    model.step = @(u, dt, x) x + dt*u;

    n_states  = 3; 
    n_actions = 3; 

    % --------------------- Cost Functions --------------------------------
    stage_cost    = @(x,u,g) stageCostPointCloud(x, u, g, ptCloud, collisionRadius, Delta_max);
    terminal_cost = @(x,u,g) norm(x - g, 2)^2;

    % --------------------- Create PLANNER Object ----------------------------
    planner = PLANNER(model, N, I, dt, lambda, sigma_eps, nu, R, ...
                n_states, n_actions, @(x,u)0, @(x,u)0 );

    % --------------------- Initial State and Path ------------------------
    x0 = [0.3; 0.5; 0.6];  % start
    path = x0;             % record trajectory

    % --------------------- Figure / Axis Initialization ------------------
    fig_bg_color = hex2rgb(colors_struct.White);
    fig = figure('Name','3D PLANNER Waypoint Planning','Color', fig_bg_color);
    ax = axes('Parent', fig);
    hold(ax, 'on'); grid(ax, 'on');
    view(ax, 3);             
    axis(ax, 'equal');      
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
    set(ax, 'XLim', [-1 8], 'YLim', [-1 5], 'ZLim', [-1 5]);

    % --------------------- 3) Show Point Cloud & Goals -------------------
    pcshow(ptCloud, 'Parent', ax, 'BackgroundColor','white');
    title(ax, 'Obstacle Point Cloud (Downsampled)', 'Color','k');

    goals_color = hex2rgb(colors_struct.RobinsEggBlue);
    plot3(ax, goals(1,:), goals(2,:), goals(3,:), 'o--',...
          'LineWidth', 2,'MarkerSize',8, 'Color', goals_color, ...
          'MarkerFaceColor', goals_color);

    for iGoal = 1:size(goals,2)
        text(goals(1,iGoal), goals(2,iGoal), goals(3,iGoal), ...
            sprintf('G%d', iGoal), 'Color','k',...
            'VerticalAlignment','bottom','HorizontalAlignment','left');
    end

    % --------------------- Main Loop: Visit Goals in Sequence ------------
    for goal_idx = 1:size(goals,2)
        current_goal = goals(:,goal_idx);
        disp(['Moving towards Goal ', num2str(goal_idx), ': (', ...
              num2str(current_goal(1)), ', ', num2str(current_goal(2)), ...
              ', ', num2str(current_goal(3)), ')']);

        while norm(x0 - current_goal) > 0.01
            planner.stage_cost    = @(x,u) stage_cost(x,u,current_goal);
            planner.terminal_cost = @(x,u) terminal_cost(x,u,current_goal);

            % PLANNER get new control
            U = planner.get_action_fmincon(x0);
            u_exec = U(:,1);

            % Step the model
            x0 = model.step(u_exec, dt, x0);
            path = [path, x0];  %#ok<AGROW>

            % -------------- Visualization of Iteration Rollouts -----------
            cla(ax);
            hold(ax, 'on'); grid(ax, 'on');
            axis(ax, 'equal');
            set(ax, 'XLim', [-1 8], 'YLim', [-1 5], 'ZLim', [-1 5]);
            xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
            view(ax, 3);

            % Redraw point cloud
            pcshow(ptCloud, 'Parent', ax, 'BackgroundColor','white');

            % Redraw goals
            plot3(ax, goals(1,:), goals(2,:), goals(3,:), 'o--',...
                  'LineWidth',2,'MarkerSize',8, 'Color', goals_color, ...
                  'MarkerFaceColor', goals_color);
            for iG = 1:size(goals,2)
                text(goals(1,iG), goals(2,iG), goals(3,iG), ...
                    sprintf('G%d', iG), 'Color','k',...
                    'VerticalAlignment','bottom','HorizontalAlignment','left');
            end

            % Plot path so far
            path_color = hex2rgb(colors_struct.Black);
            plot3(ax, path(1,:), path(2,:), path(3,:), '.-', ...
                  'LineWidth',1.5, 'Color', path_color);

            % Current target
            plot3(ax, current_goal(1), current_goal(2), current_goal(3), ...
                  'o','MarkerSize',8, 'MarkerFaceColor', goals_color, 'MarkerEdgeColor','k');

            drawnow;
        end
    end
    disp('Reached all goals!');

    % === At this point, "path" holds the final 3D trajectory. ===

    % -------------------------
    % 1) Load the Panda robot model
    % -------------------------
    robot = loadrobot("frankaEmikaPanda","DataFormat","row");

    % 2) Inverse Kinematics setup
    ik = inverseKinematics('RigidBodyTree', robot);
    endEffectorName = 'panda_link8';
    weights         = [1, 1, 1, 1, 1, 1];
    initialGuess    = robot.homeConfiguration;
    
    % Desired orientation for the end effector
    desiredOrientation = eul2tform([-pi/2, 0, 0], "XYZ");
    
    % 3) Solve IK for each waypoint in "path"
    configurations = [];
    for i = 1:size(path,2)
        % Optionally shift your waypoints if needed
        p = path(:,i) - [0; 0.15; 0];  
        tform = trvec2tform(p') * desiredOrientation;
    
        [configSol, solInfo] = ik(endEffectorName, tform, weights, initialGuess);
        configurations = [configurations; configSol];
        initialGuess   = configSol;
    end

    % 4) Interpolate for smoother motion
    nInterp = 10;
    interpConfigs = [];
    for i = 1:size(configurations,1)-1
        cfg1 = configurations(i,:);
        cfg2 = configurations(i+1,:);
        for j = 0:nInterp-1
            alpha = j/(nInterp-1);
            cfg   = (1 - alpha)*cfg1 + alpha*cfg2;
            interpConfigs = [interpConfigs; cfg];
        end
    end
    interpConfigs = [interpConfigs; configurations(end,:)];

    % -------------------------
    % 5) Preview (not recorded)
    % -------------------------
    figure('Color','white','Name','Preview Figure');
    axPreview = axes('Parent', gcf);
    hold(axPreview, 'on'); grid(axPreview, 'on');
    axis(axPreview, 'equal');
    xlabel(axPreview,'X'); ylabel(axPreview,'Y'); zlabel(axPreview,'Z');
    view(axPreview,3);
    
    pcshow(ptCloud, 'Parent', axPreview, 'BackgroundColor','white');
    camlight('headlight');
    lighting gouraud;  
    material dull;
    
    % Plot final path
    plot3(axPreview, path(1,:), path(2,:), path(3,:), '.-','Color','k','LineWidth',1.5);
    
    % Plot goals
    plot3(axPreview, goals(1,:), goals(2,:), goals(3,:), 'o--','LineWidth',2, ...
          'MarkerFaceColor',goals_color,'MarkerSize',8);
    
    show(robot, interpConfigs(1,:), 'Parent', axPreview, ...
         'PreservePlot', false, 'FastUpdate', true);
    view(axPreview, [45 45]);
    title(axPreview, 'Preview: Robot + Point Cloud + Path', 'Color','k');

    % -------------------------
    % 6) Create VideoWriter
    % -------------------------
    fps = 30;
    writerObj = VideoWriter('panda_motion');
    writerObj.FrameRate = fps;
    open(writerObj);

    % -------------------------
    % 7) Create figure with fixed resolution for recording
    % -------------------------
    vidWidth  = 840;
    vidHeight = 630;
    figVideo = figure('Name','Recording Figure',...
        'Units','pixels','Position',[100 100 vidWidth vidHeight],...
        'Color','white',...
        'MenuBar','none',...
        'ToolBar','none',...
        'Resize','off',...
        'WindowStyle','normal');
    drawnow;  % force figure to update size
    
    axVideo = axes('Parent', figVideo, 'Units','pixels',...
                   'Position',[0 0 vidWidth vidHeight], 'Color','white');
    hold(axVideo, 'on');
    grid(axVideo, 'on');
    axis(axVideo,'equal');
    xlabel(axVideo, 'X'); ylabel(axVideo, 'Y'); zlabel(axVideo, 'Z');
    
    pcshow(ptCloud, 'Parent', axVideo, 'BackgroundColor','white');
    camlight('headlight');
    lighting gouraud;  
    material dull;
    
    % ---- Plot the same path & goals in the final video figure ----
    plot3(axVideo, path(1,:), path(2,:), path(3,:), '.-','Color','k','LineWidth',1.5);
    
    plot3(axVideo, goals(1,:), goals(2,:), goals(3,:), 'o--',...
          'LineWidth',2, 'MarkerSize',8, 'Color','Black','MarkerFaceColor',goals_color);
    for iGoal = 1:size(goals,2)
        text(goals(1,iGoal), goals(2,iGoal), goals(3,iGoal), ...
             sprintf('G%d', iGoal), 'Color','Black',...
             'VerticalAlignment','top','HorizontalAlignment','left', ...
             'Parent', axVideo);
    end

    view(axVideo, [45 45]);
    title(axVideo, 'Recording: Robot + Point Cloud + Path','Color','k');
    
    % -------------------------
    % 8) Animate robot & capture frames
    % -------------------------
    for i = 1:size(interpConfigs,1)
        show(robot, interpConfigs(i,:), 'Parent', axVideo, ...
             'PreservePlot', false, 'FastUpdate', true);
        drawnow;

        % Force capture of the exact 840x630 region:
        frame = getframe(figVideo, [0 0 vidWidth vidHeight]);
        writeVideo(writerObj, frame);
    end

    close(writerObj);
    disp('Animation video saved successfully!');
end

% ---------------------- Stage Cost Function (Point Cloud) ---------------
function c = stageCostPointCloud(x, u, current_goal, ptCloud, collisionRadius, Delta_max)
    dist_goal = norm(x - current_goal, 2);
    
    [~, dists] = findNearestNeighbors(ptCloud, x', 1);
    nearestDist = dists(1);
    if nearestDist < collisionRadius
        penalty_obstacle = 1e2;
    else
        penalty_obstacle = 0;
    end

    % Soft constraint on maximum step distance
    dt = 0.2;  % If you want it to match dt=0.5, set dt=0.5
    step_size = norm(u,2)*dt;
    if step_size > Delta_max
        penalty_step = 1e3 * (step_size - Delta_max)^2;
    else
        penalty_step = 0;
    end

    % Control usage penalty
    penalty_control = 0.1 * (u.' * u);

    % Combine stage cost
    c = 100*dist_goal^2 + penalty_obstacle + penalty_step + penalty_control;
end

% ---------------------- Hex to RGB Helper -------------------------------
function rgb = hex2rgb(hexStr)
    hexStr = strrep(hexStr,'#','');
    if length(hexStr) ~= 6
        error('hex color code must be 6 characters, e.g. "#FF00FF"');
    end
    r = hex2dec(hexStr(1:2)) / 255;
    g = hex2dec(hexStr(3:4)) / 255;
    b = hex2dec(hexStr(5:6)) / 255;
    rgb = [r, g, b];
end
