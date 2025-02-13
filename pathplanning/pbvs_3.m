function example_3D_planning_with_pointcloud_PBVS()
    clear; clc; close all;

    %% ------------------------ Define Color Scheme ------------------------
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

    %% --------------------- 1) Load and Downsample the Point Cloud -------
    ptCloud = pcread('vine_simple.pcd');
    voxelSize = 0.02;  % Define the size of each voxel
    ptCloud = pcdownsample(ptCloud, 'gridAverage', voxelSize);
    
    collisionRadius = 0.01;  

    %% --------------------- 2) 3D Environment & Problem Setup ------------
    % Define position and orientation goals.
    % Each column is a goal pose [x; y; z; roll; pitch; yaw]
    goals_poses = [
        0.23958,       0.43975;   % x-coords
        0.6,           0.6;       % y-coords
        0.6098,      0.65;      % z-coords
        0,                 0;         % roll
        0,                0;         % pitch
        pi/2,           pi/2       % yaw
    ];
    num_goals = size(goals_poses,2);
    goals = goals_poses;

    % (The following Delta_max was previously used as a single bound.)
    % Now we define separate bounds for position and orientation:
    Delta_max_pos = 0.025;     % Maximum allowed position change per step
    Delta_max_orient = 1; % Maximum allowed orientation change (in radians) per step

    %% --------------------- PLANNER Hyperparameters -----------------------
    dt        = 0.05;            
    N         = 5;              
    I         = 10;           
    lambda    = 0.5;            
    sigma_eps = (1e-2^2)*eye(6);  % Updated for 6D state
    nu        = 1.0;            
    R         = 0.1*eye(6);     % Updated for 6D

    %% --------------------- Model Definition ------------------------------
    % State: [x; y; z; roll; pitch; yaw]
    model = struct();
    model.step = @(u, dt, x) x + u;   % Simple additive (Euler) integration

    n_states  = 6; 
    n_actions = 6; 

    %% --------------------- Cost Functions --------------------------------
    % Use homogeneous error for pose error plus a visibility reward.
    stage_cost = @(x,u,g) stageCostPose(x, u, g, ptCloud, collisionRadius);
    terminal_cost = @(x,u,g) terminalCostPose(x, g);
    
    %% --------------------- Create PLANNER Object -------------------------
    % Pass the new bounds to the planner.
    planner = PLANNER(model, N, I, dt, lambda, sigma_eps, nu, R, ...
                n_states, n_actions, @(x,u)0, @(x,u)0, Delta_max_pos, Delta_max_orient );

    %% --------------------- Initial State and Path ------------------------
    % Start at a given position with zero orientation.
    x = [0.0; 0.5; 0.6; 0; 0; 0];  

    %% --------------------- Figure / Axis Initialization ------------------
    fig_bg_color = hex2rgb(colors_struct.White);
    fig = figure('Name','3D PLANNER Waypoint Planning','Color', fig_bg_color);
    ax = axes('Parent', fig);
    hold(ax, 'on'); grid(ax, 'on');
    view(ax, 3);             
    axis(ax, 'equal');      
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
    set(ax, 'XLim', [-1 8], 'YLim', [-1 5], 'ZLim', [-1 5]);

    %% --------------------- 3) Show Point Cloud & Goals -------------------
    pcshow(ptCloud, 'Parent', ax, 'BackgroundColor','white');
    title(ax, 'Obstacle Point Cloud (Downsampled)', 'Color','k');

    goals_color = hex2rgb(colors_struct.RobinsEggBlue);
    % Plot only the positions of the goals.
    plot3(ax, goals(1,:), goals(2,:), goals(3,:), 'o--',...
          'LineWidth', 2,'MarkerSize',8, 'Color', goals_color, ...
          'MarkerFaceColor', goals_color);
    for iGoal = 1:size(goals,2)
        text(goals(1,iGoal), goals(2,iGoal), goals(3,iGoal), ...
            sprintf('G%d', iGoal), 'Color','k',...
            'VerticalAlignment','bottom','HorizontalAlignment','left');
    end

    %% --------------------- Main Loop: Visit Goals in Sequence ------------
    camera_poses = [];  % Will store poses as [x y z roll pitch yaw]
    
    for goal_idx = 1:size(goals,2)
        current_goal = goals(:,goal_idx);
        disp(['Moving towards Goal ', num2str(goal_idx), ': (', ...
              num2str(current_goal(1)), ', ', num2str(current_goal(2)), ...
              ', ', num2str(current_goal(3)), ') with orientation (', ...
              num2str(current_goal(4)), ', ', num2str(current_goal(5)), ', ', num2str(current_goal(6)), ')']);
    
        % Compute the current and goal transforms.
        H_x = trvec2tform(x(1:3)') * eul2tform(x(4:6)', 'XYZ');
        H_goal = trvec2tform(current_goal(1:3)') * eul2tform(current_goal(4:6)', 'XYZ');
    
        while norm(homogeneous_error(H_goal, H_x)) > 0.01
            % Update cost functions with the current goal.
            planner.stage_cost    = @(x,u) stage_cost(x, u, current_goal);
            planner.terminal_cost = @(x,u) terminal_cost(x, u, current_goal);
    
            % Get new control from the planner.
            U = planner.get_action(x);
            u_exec = U(:,1);
    
            % Step the “robot” in 6D (position and orientation)
            x = model.step(u_exec, dt, x);
    
            % Update the current transform.
            H_x = trvec2tform(x(1:3)') * eul2tform(x(4:6)', 'XYZ');
    
            % Save the current pose.
            camera_poses = [camera_poses; x'];
    
            % -------------- Visualization of Iteration Rollouts -----------
            cla(ax);
            hold(ax, 'on'); grid(ax, 'on');
            axis(ax, 'equal');
            set(ax, 'XLim', [-1 8], 'YLim', [-1 5], 'ZLim', [-1 5]);
            xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
            view(ax, 3);
    
            % Redraw the point cloud.
            pcshow(ptCloud, 'Parent', ax, 'BackgroundColor','white');
    
            % Redraw the goals.
            plot3(ax, goals(1,:), goals(2,:), goals(3,:), 'o--',...
                  'LineWidth',2,'MarkerSize',8, 'Color', goals_color, ...
                  'MarkerFaceColor', goals_color);
            for iG = 1:size(goals,2)
                text(goals(1,iG), goals(2,iG), goals(3,iG), ...
                    sprintf('G%d', iG), 'Color','k',...
                    'VerticalAlignment','bottom','HorizontalAlignment','left');
            end
    
            % Plot the path so far (using positions only).
            path_color = hex2rgb(colors_struct.Black);
            plot3(ax, camera_poses(:,1), camera_poses(:,2), camera_poses(:,3), '.-', ...
                  'LineWidth',1.5, 'Color', path_color);
    
            % Mark the current target.
            plot3(ax, current_goal(1), current_goal(2), current_goal(3), ...
                  'o','MarkerSize',8, 'MarkerFaceColor', goals_color, 'MarkerEdgeColor','k');
    
            % Visualize the camera frustum using the current pose.
            visualizeCameraFrustum(x(1:3), x(4:6), ptCloud, ax);
    
            drawnow;
        end
    end
    disp('Reached all goals!');

     %% -------------- NEW: Assess Visibility Performance --------------
    % Loop through each planned camera pose, count how many points were seen 
    % (using the same FOV, range, and transformation as in the cost term),
    % and sum these counts to assess the overall visibility performance.
    totalVisiblePoints = 0;
    fov = 60; 
    max_range = 0.5; 
    min_range = 0.0;
    for i = 1:size(camera_poses,1)
        x_pose = camera_poses(i,:)';
        Rwc = eul2rotm(x_pose(4:6)', 'XYZ');
        ptCloud_camera = transformPointCloud(ptCloud, x_pose(1:3), Rwc);
        visible_bool = isPointInFOV(ptCloud_camera, fov, max_range, min_range);
        count_visible = sum(visible_bool);
        totalVisiblePoints = totalVisiblePoints + count_visible;
    end
    disp(['Total number of points seen along the planned path: ', num2str(totalVisiblePoints)]);
    averageVisiblePoints = totalVisiblePoints / size(camera_poses,1);
    disp(['Average number of points seen per pose along the planned path: ', num2str(averageVisiblePoints)]);

    %% ======================= POSITION-BASED VISUAL SERVOING =======================
    % The planned poses are stored in 'camera_poses' as [x y z roll pitch yaw].
    % Convert these poses into homogeneous transforms.
    pathTransforms = cell(size(camera_poses,1), 1);
    for i = 1:size(camera_poses,1)
        p = camera_poses(i,1:3);
        eulAngles = camera_poses(i,4:6); % [roll, pitch, yaw]
        R_mat = eul2rotm(eulAngles, 'XYZ');
        pathTransforms{i} = trvec2tform(p) * rotm2tform(R_mat);
    end

    % 2) Load the Panda robot model.
    robot = loadrobot("frankaEmikaPanda","DataFormat","row");
    endEffectorName = 'panda_hand';

    % --- Initialize the robot near the first transform in pathTransforms.
    T_first = pathTransforms{1};
    T_extraRot = eul2tform([0, pi/2, 0], "XYZ"); % extra rotation if desired
    T_start = T_first * T_extraRot;

    ik = inverseKinematics('RigidBodyTree', robot);
    weights = [1, 1, 1, 1, 1, 1];
    initialGuess = robot.homeConfiguration;
    [configSol, solInfo] = ik(endEffectorName, T_start, weights, initialGuess);
    q = configSol;  

    %% --------------------- PBVS Controller Parameters ----------------------
    Kp_r = 5.0;    % rotational gain
    Kp_t = 40.0;   % translational gain
    % The homogeneous_error returns a 6-vector where the first 3 elements are the orientation error.
    K = diag([Kp_r, Kp_r, Kp_r, Kp_t, Kp_t, Kp_t]);

    dt_servo  = 0.01;   % servo time step
    maxIter   = 50;     % maximum iterations
    threshold = 1e-2;   % final goal threshold

    L = 1;  % Lookahead offset (path index)
    servoTrajectory = q;
    velocityHistory = [];  
    refPositions = zeros(maxIter, 3);  % To store the lookahead reference positions
    
    %% --------------------- Lookahead PBVS Loop ---------------------------
    for iter = 1:maxIter
        T_current = getTransform(robot, q, endEffectorName);
    
        T_final = pathTransforms{end} * T_extraRot;
        err_final = homogeneous_error(T_final, T_current);
        if norm(err_final) < threshold
            disp('Lookahead PBVS: Reached final pose (within threshold).');
            refPositions = refPositions(1:iter,:);
            break;
        end
    
        kClosest = findClosestPoseOnPath(T_current, pathTransforms);
        kLA = min(kClosest + L, length(pathTransforms));
        T_ref = pathTransforms{kLA} * T_extraRot;
    
        refPositions(iter,:) = T_ref(1:3,4)';
    
        err_vec = homogeneous_error(T_ref, T_current);
        v = K * err_vec;   % 6x1 twist: [omega_x; omega_y; omega_z; v_x; v_y; v_z]
        velocityHistory = [velocityHistory, v];
    
        J = geometricJacobian(robot, q, endEffectorName);
        q_dot = pinv(J) * v;
        q = q + (q_dot' * dt_servo);
    
        % if iter == 200
        %     disp('>>> Applying external disturbance to the robot joints! <<<');
        %     q = q + 0.05*randn(size(q));  
        % end
    
        servoTrajectory = [servoTrajectory; q];
    end

    finalIter = size(servoTrajectory,1);
    if finalIter < size(refPositions,1)
        refPositions = refPositions(1:finalIter,:);
    end

    %% --------------------- Plot the Twist (v) ----------------------------
    nSamples = size(velocityHistory, 2);
    timeVec = (0:(nSamples-1)) * dt_servo;
    
    figure('Name','End-Effector Twist vs. Time');
    subplot(2,1,1);
    plot(timeVec, velocityHistory(1,:), 'LineWidth',1.5); hold on;
    plot(timeVec, velocityHistory(2,:), 'LineWidth',1.5);
    plot(timeVec, velocityHistory(3,:), 'LineWidth',1.5);
    legend('\omega_x','\omega_y','\omega_z','Location','Best');
    xlabel('Time [s]'); ylabel('Angular Velocity [rad/s]');
    title('End-Effector Angular Velocity (PBVS)');
    grid on;
    
    subplot(2,1,2);
    plot(timeVec, velocityHistory(4,:), 'LineWidth',1.5); hold on;
    plot(timeVec, velocityHistory(5,:), 'LineWidth',1.5);
    plot(timeVec, velocityHistory(6,:), 'LineWidth',1.5);
    legend('v_x','v_y','v_z','Location','Best');
    xlabel('Time [s]'); ylabel('Linear Velocity [m/s]');
    title('End-Effector Linear Velocity (PBVS)');
    grid on;

    %% =================== ANIMATE THE ROBOT MOTION (AND RECORD VIDEO) ===================
    fps = 60;
    writerObj = VideoWriter('panda_motion_PBVS_lookahead');
    writerObj.FrameRate = fps;
    open(writerObj);

    vidWidth  = 840;
    vidHeight = 630;
    figVideo = figure('Name','Recording Figure',...
        'Units','pixels','Position',[100 100 vidWidth vidHeight],...
        'Color','black',...
        'MenuBar','none',...
        'ToolBar','none',...
        'Resize','off',...
        'WindowStyle','normal');
    drawnow;

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
    view(axVideo, [20 45]);
    title(axVideo, 'Recording: Robot + Point Cloud + Path (Lookahead PBVS)','Color','k');

    pathPositions = zeros(length(pathTransforms),3);
    for iP = 1:length(pathTransforms)
        pathPositions(iP,:) = pathTransforms{iP}(1:3,4)';
    end

    numFrames = size(servoTrajectory,1);
    for i = 1:numFrames
        cla(axVideo);
        plot3(axVideo, pathPositions(:,1), pathPositions(:,2), pathPositions(:,3), ...
              '--','Color','red','LineWidth',1.5);
        plot3(axVideo, goals(1,:), goals(2,:), goals(3,:), 'o--',...
              'LineWidth',2, 'MarkerSize',8, 'Color','Red','MarkerFaceColor','black');
        for iGoal = 1:size(goals,2)
            text(goals(1,iGoal), goals(2,iGoal), goals(3,iGoal), ...
                 sprintf('G%d', iGoal), 'Color','White',...
                 'VerticalAlignment','top','HorizontalAlignment','left', ...
                 'Parent', axVideo);
        end

        pcshow(ptCloud, 'Parent', axVideo, 'BackgroundColor','black');
        camlight('headlight');
        lighting gouraud;  
        material dull;

        show(robot, servoTrajectory(i,:), 'Parent', axVideo, ...
             'PreservePlot', false, 'FastUpdate', true, Visuals="on");
    
        tform_ee = getTransform(robot, servoTrajectory(i,:), endEffectorName);
        pos = tform_ee(1:3,4);
        eul_angles = rotm2eul(tform_ee(1:3,1:3)*eul2rotm([0,-pi/2,0], "XYZ"), 'XYZ');
        visualizeCameraFrustum(pos, eul_angles, ptCloud, axVideo);

        if i <= size(refPositions,1)
            refPos = refPositions(i,:);
            plot3(axVideo, refPos(1), refPos(2), refPos(3), 'mo', ...
                  'MarkerSize',10, 'LineWidth',1.5, 'MarkerFaceColor','m');
        end
    
        drawnow;
        frame = getframe(figVideo, [0 0 vidWidth vidHeight]);
        writeVideo(writerObj, frame);
    end

    close(writerObj);
    disp('Video recording complete.');
    
    %% =================== PLOT SERVO TRAJECTORY ===================
    numPoints = size(servoTrajectory, 1);
    ee_positions = zeros(numPoints, 3);
    for i = 1:numPoints
        T = getTransform(robot, servoTrajectory(i,:), endEffectorName);
        ee_positions(i,:) = T(1:3,4)';
    end
    
    figure;
    plot3(ee_positions(:,1), ee_positions(:,2), ee_positions(:,3), 'b.-', 'LineWidth', 1.5);
    hold on;
    plot3(camera_poses(:,1), camera_poses(:,2), camera_poses(:,3), 'r.--', 'LineWidth', 2);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    legend('Servo Trajectory', 'Planned Path', 'Location', 'Best');
    title('End-Effector Servo Trajectory vs. Planned Path (Lookahead PBVS)');
    grid on;
    axis equal;
end

%% ================== HELPER FUNCTIONS ==================

function kClosest = findClosestPoseOnPath(H_current, pathTransforms)
    pCurrent = H_current(1:3,4);
    minDist = inf;
    kClosest = 1;
    for i = 1:length(pathTransforms)
        p_i = pathTransforms{i}(1:3,4);
        d = norm(p_i - pCurrent);
        if d < minDist
            minDist = d;
            kClosest = i;
        end
    end
end

function c = stageCostPose(x, u, current_goal, ptCloud, collisionRadius)
    % Convert the state and goal into homogeneous transforms.
    H_x = trvec2tform(x(1:3)') * eul2tform(x(4:6)', 'XYZ');
    H_goal = trvec2tform(current_goal(1:3)') * eul2tform(current_goal(4:6)', 'XYZ');
    err = homogeneous_error(H_goal, H_x);
    
    % Weight the orientation and translation errors.
    W = diag([10, 10, 10, 100, 100, 100]);
    cost_pose = err' * W * err;
    
    % Add a penalty if the position is too close to an obstacle.
    [~, dists] = findNearestNeighbors(ptCloud, x(1:3)', 1);
    nearestDist = dists(1);
    if nearestDist < collisionRadius
        penalty_obstacle = 1e2;
    else
        penalty_obstacle = 0;
    end

    % Control effort penalty.
    penalty_control = 0; %0.1 * (u.' * u);
    
    % ---------------------- New: Visibility Reward ----------------------
    % This term rewards (i.e. subtracts cost) for viewing more of the point cloud.
    % It is scaled by the distance-to-goal so that near the cut point the term fades.
    fov = 60; 
    max_range = 0.3; 
    min_range = 0.0; 
    Rwc = eul2rotm(x(4:6)', 'XYZ');
    ptCloud_camera = transformPointCloud(ptCloud, x(1:3), Rwc);
    visible_bool = isPointInFOV(ptCloud_camera, fov, max_range, min_range);
    count_visible = sum(visible_bool);

    % Scale the reward based on distance to goal (diminish near the goal).
    d = norm(x(1:3) - current_goal(1:3));
    d_thresh = 0.1; % if d < 0.2, scale < 1.
    scale = min(1, d / d_thresh);

    w_visibility = 10;  % Tuning parameter (adjust as needed)
    % cost_visibility = w_visibility * (1 - count_visible / ptCloud.Count)*scale;
    alpha = 5; % tuning parameter to control the saturation rate
    cost_visibility = w_visibility * exp(-alpha * (count_visible / ptCloud.Count))*scale
    % ---------------------------------------------------------------------
    c = cost_pose + penalty_obstacle + penalty_control + double(cost_visibility);
end

function c = terminalCostPose(x, current_goal)
    H_x = trvec2tform(x(1:3)') * eul2tform(x(4:6)', 'XYZ');
    H_goal = trvec2tform(current_goal(1:3)') * eul2tform(current_goal(4:6)', 'XYZ');
    c = norm(homogeneous_error(H_goal, H_x))^2;
end

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

function visualizeCameraFrustum(position, orientation, ptCloud, ax)
    fov = 60; 
    max_range = 0.3; 
    min_range = 0.0; 

    Rwc = eul2rotm(reshape(orientation,[1,3]), 'XYZ');
    ptCloud_camera = transformPointCloud(ptCloud, position, Rwc);

    in_fov = isPointInFOV(ptCloud_camera, fov, max_range, min_range);
    visible_points = ptCloud.Location(in_fov, :);

    plot3(ax, visible_points(:, 1), visible_points(:, 2), visible_points(:, 3), ...
          'g.', 'MarkerSize', 10);
    plot3(ax, position(1), position(2), position(3), 'ro', ...
          'MarkerSize', 10, 'MarkerFaceColor', 'r');

    plotFrustumEdges(position, orientation, fov, max_range, ax);
end

function ptCloud_camera = transformPointCloud(ptCloud, position, Rwc)
    points = ptCloud.Location;
    rCWw = position;
    points_camera = (Rwc.' * (points' - rCWw))';
    ptCloud_camera = pointCloud(points_camera);
end

function in_fov = isPointInFOV(ptCloud_camera, fov, max_range, min_range)
    points = ptCloud_camera.Location;
    x = points(:, 1);
    y = points(:, 2);
    z = points(:, 3);

    in_front = x > 0;
    angles_x = atan2d(y, x); 
    angles_y = atan2d(z, x); 

    in_fov_x = abs(angles_x) <= fov / 2; 
    in_fov_y = abs(angles_y) <= fov / 2; 
    in_range = (x >= min_range) & (x <= max_range);

    in_fov = in_front & in_fov_x & in_fov_y & in_range;
end

function plotFrustumEdges(position, orientation, fov, max_range, ax)
    fov_rad = deg2rad(fov);
    half_width = max_range * tan(fov_rad / 2);
    half_height = half_width;

    frustum_corners_camera = [
        0, 0, 0;
        max_range, -half_width, -half_height;
        max_range, -half_width,  half_height;
        max_range,  half_width,  half_height;
        max_range,  half_width, -half_height
    ];

    Rwc = eul2rotm(reshape(orientation,[1,3]), 'XYZ');
    frustum_corners_world = (Rwc * frustum_corners_camera')' + position';

    light_grey = [0.7, 0.7, 0.7];

    edges = [
        1 2; 1 3; 1 4; 1 5;
        2 3; 3 4; 4 5; 5 2
    ];
    for eIdx = 1:size(edges,1)
        idxA = edges(eIdx,1); 
        idxB = edges(eIdx,2);
        plot3(ax, ...
            [frustum_corners_world(idxA,1), frustum_corners_world(idxB,1)], ...
            [frustum_corners_world(idxA,2), frustum_corners_world(idxB,2)], ...
            [frustum_corners_world(idxA,3), frustum_corners_world(idxB,3)], ...
            'Color', light_grey, 'LineWidth', 1.5);
    end
end

function e = homogeneous_error(H1, H2)
    e = zeros(6,1);
    R1 = H1(1:3, 1:3);
    R2 = H2(1:3, 1:3);
    Re = R1 * R2';
    t = trace(Re);
    eps_vec = [Re(3,2) - Re(2,3);
               Re(1,3) - Re(3,1);
               Re(2,1) - Re(1,2)];
    eps_norm = norm(eps_vec);
    
    if (t > -0.99 || eps_norm > 1e-10)
        if eps_norm < 1e-3
            orientation_error = (0.75 - t/12) * eps_vec;
        else
            orientation_error = (atan2(eps_norm, t - 1) / eps_norm) * eps_vec;
        end
    else
        orientation_error = (pi/2) * (diag(Re) + 1);
    end
    
    e(1:3) = orientation_error; 
    e(4:6) = H1(1:3,4) - H2(1:3,4);
end