function [] = known_large_MCT(seed, horizon, k_factor,root, mct, num_mct) 

rng(str2double(seed))
rng
%file_name = strcat(seed,'_hor',horizon,'.txt');
path = [root '/rsmith/lab-members/nhakimi/known_model_v2_2024/results/'];
file_name = strcat(path,  horizon,'hor_',k_factor,'kfactor_', mct, 'MCT_', num_mct, 'num_mct_', seed,'seed_survival_time','.mat');
matfilename = strcat(path, horizon,'hor_',k_factor,'kfactor_',mct, 'MCT_', num_mct, 'num_mct_',seed,'seed','.mat');

len_each = 1;

k_factor = str2double(k_factor);
horizon = str2double(horizon);
mct = str2double(mct);
num_mct = str2double(num_mct);
%fid = fopen(file_name, 'w');

previous_positions = [];

hill_1 = 55;
true_food_source_1 = 71;
true_food_source_2 = 43;
true_food_source_3 =57;
true_food_source_4 = 78;
true_water_source_1 = 73;
true_water_source_2 = 33;
true_water_source_3 = 48;
true_water_source_4 = 67;
true_sleep_source_1 = 64;
true_sleep_source_2 = 44;
true_sleep_source_3 = 49;
true_sleep_source_4 = 59;

num_states = 100;
num_states_low = 25;
% why number num_states_low = 25;

A{1}(:,:,:) = zeros(num_states,num_states,4);
a{1}(:,:,:) = zeros(num_states,num_states,4);
for i = 1:num_states
    A{1}(i,i,:) = 1;
    a{1}(i,i,:) = 1;
end

A{2}(:,:,:) = zeros(2,num_states,4); 
A{2}(1,:,:) = 1;
A{2}(2,true_food_source_1,1) = 1;
A{2}(1,true_food_source_1,1) = 0;
A{2}(2,true_food_source_2,2) = 1;
A{2}(1,true_food_source_2,2) = 0;
A{2}(2,true_food_source_3,3) = 1;
A{2}(1,true_food_source_3,3) = 0;
A{2}(2,true_food_source_4,4) = 1;
A{2}(1,true_food_source_4,4) = 0;
A{2}(3,true_water_source_1,1) = 1;
A{2}(1,true_water_source_1,1) = 0;
A{2}(3,true_water_source_2,2) = 1;
A{2}(1,true_water_source_2,2) = 0;
A{2}(3,true_water_source_3,3) = 1;
A{2}(1,true_water_source_3,3) = 0;
A{2}(3,true_water_source_4,4) = 1;
A{2}(1,true_water_source_4,4) = 0;
A{2}(4,true_sleep_source_1,1) = 1;
A{2}(1,true_sleep_source_1,1) = 0;
A{2}(4,true_sleep_source_2,2) = 1;
A{2}(1,true_sleep_source_2,2) = 0;
A{2}(4,true_sleep_source_3,3) = 1;
A{2}(1,true_sleep_source_3,3) = 0;
A{2}(4,true_sleep_source_4,4) = 1;
A{2}(1,true_sleep_source_4,4) = 0;
A{3}(:,:,:) = zeros(5,num_states,4);
A{3}(5,:,:) = 1;
A{3}(1,hill_1,1) = 1;
A{3}(5,hill_1,1) = 0;
A{3}(2,hill_1,2) = 1;
A{3}(5,hill_1,2) = 0;
A{3}(3,hill_1,3) = 1;
A{3}(5,hill_1,3) = 0;
A{3}(4,hill_1,4) = 1;
A{3}(5,hill_1,4) = 0;
a{3} = A{3};
a{2}(:,:,:) = zeros(4,num_states,4);
a{2} = a{2} + 0.1;
a{2} = A{2};


%% Setup Agent

D{1} = zeros(1,num_states)'; %position in environment
D{2} = [0.25,0.25,0.25,0.25]';

D{1}(51) = 1; % start position


survival(:) = zeros(1,70);

%why these? wtf is happening?

D{1} = normalise(D{1});
num_factors = 1;
T = 27; % what is this

num_modalities = 3;
num_states = 100;
food_locations = [true_food_source_1, true_food_source_2, true_food_source_3] ;
resource_locations = [food_locations];
short_term_memory(:,:,:,:,:) = zeros(40,40,40,400);

%%% Distributions %%%

for action = 1:5
    B{1}(:,:,action)  =  eye(num_states);
    B{2}(:,:,action)  =  zeros(4);
    B{2}(:,:,action) = [0.95,   0,     0     0.05;
               0.05,   0.95,   0,    0;
               0,     0.05,   0.95,  0;
               0,     0,     0.05   0.95 ];
    
end
b = B;
for i = 1:num_states
    if i ~= [1,11,21,31,41,51,61,71,81,91]
        B{1}(:,i,2) = circshift(B{1}(:,i,2),-1); % move left
    end  
end

for i = 1:num_states
    if i ~= [10,20,30,40,50,60,70,80,90]
        B{1}(:,i,3) = circshift(B{1}(:,i,3),1); % move right
    end  
end

for i = 1:num_states
    if i ~= [91,92,93,94,95,96,97,98,99,100]
        B{1}(:,i,4) = circshift(B{1}(:,i,4),10); % move rup
    end  
end

for i = 1:num_states
    if i ~= [1,2,3,4,5,6,7,8,9,10]
        B{1}(:,i,5) = circshift(B{1}(:,i,5),-10); % move down
    end  
end

b{1} = B{1};

time_since_food = 0;    
time_since_water = 0;
time_since_sleep = 0;
t = 1;
surety = 1;
simulated_time = 0;
action_history = zeros(100,99);
states_history = zeros(100,99);
season_history = zeros(100,99);

global TREE_SEARCH_HISTORY;
TREE_SEARCH_HISTORY = cell(100, 99);
survival_times=[];
%% Main Loop
for trial = 1:len_each
    chosen_action = zeros(1,99);
while(t<100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
    disp(t)
    bb{2} = normalise_matrix(b{2}); % what is the difference between b and bb
    for factor = 1:2
        if t == 1
            P{t,factor} = D{factor}';
            Q{t,factor} = D{factor}';
            true_states{trial}(1, t) = 51;
            true_states{trial}(2, t) = find(cumsum(D{2}) >= rand,1);
        else  
            if factor == 1
                Q{t,factor} = (B{1}(:,:,chosen_action(t-1))*Q{t-1,factor}')';
                true_states{trial}(factor, t) = find(cumsum(B{1}(:,true_states{trial}(factor,t-1),chosen_action(t-1)))>= rand,1);
            else
                %b = B{2}(:,:,:);
                Q{t,factor} = (bb{2}(:,:,chosen_action(t-1))*Q{t-1,factor}')';%(B{2}(:,:)'
                true_states{trial}(factor, t) = find(cumsum(B{2}(:,true_states{trial}(factor,t-1),1))>= rand,1);   
                 
            end
        end
    states_history(trial, t) = true_states{trial}(1, t);
    season_history(trial, t) = true_states{trial}(2, t);      
    end
    
    if (true_states{trial}(2,t) == 1 && true_states{trial}(1,t) == true_food_source_1) || (true_states{trial}(2,t) == 2 && true_states{trial}(1,t) == true_food_source_2) || (true_states{trial}(2,t) == 3 && true_states{trial}(1,t) == true_food_source_3) || (true_states{trial}(2,t) == 4 && true_states{trial}(1,t) == true_food_source_4)
            time_since_food = 0;
            time_since_water = time_since_water +1;
            time_since_sleep = time_since_sleep +1;
                       
    elseif (true_states{trial}(2,t) == 1 && true_states{trial}(1,t) == true_water_source_1) || (true_states{trial}(2,t) == 2 && true_states{trial}(1,t) == true_water_source_2) || (true_states{trial}(2,t) == 3 && true_states{trial}(1,t) == true_water_source_3) || (true_states{trial}(2,t) == 4 && true_states{trial}(1,t) == true_water_source_4)
        time_since_water = 0;
        time_since_food = time_since_food +1;
        time_since_sleep = time_since_sleep +1;

    elseif (true_states{trial}(2,t) == 1 && true_states{trial}(1,t) == true_sleep_source_1) || (true_states{trial}(2,t) == 2 && true_states{trial}(1,t) == true_sleep_source_2) || (true_states{trial}(2,t) == 3 && true_states{trial}(1,t) == true_sleep_source_3) || (true_states{trial}(2,t) == 4 && true_states{trial}(1,t) == true_sleep_source_4)
        time_since_sleep = 0;
        time_since_food = time_since_food +1;
        time_since_water = time_since_water +1;
      
    else
        if t > 1
            time_since_food = time_since_food +1;
             time_since_water = time_since_water +1;
             time_since_sleep = time_since_sleep +1;
        end

    end
    % sample the next observation. Same technique as sampling states
    
    for modality = 1:num_modalities     
        ob = A{modality}(:,true_states{trial}(1,t),true_states{trial}(2,t));
        observations(modality,t) = find(cumsum(A{modality}(:,true_states{trial}(1,t),true_states{trial}(2,t)))>=rand,1);
        %create a temporary vectore of 0s
        vec = zeros(1,size(A{modality},1));
        % set the index of the vector matching the observation index to 1
        vec(1,observations(modality,t)) = 1;
        O{modality,t} = vec;
    end
    true_t = t;
         
      if true_states{trial}(2,t) == 1
           food = true_food_source_1;
           water = true_water_source_1;
           sleep = true_sleep_source_1;
      elseif true_states{trial}(2,t) == 2
          food = true_food_source_2;
           water = true_water_source_2;
           sleep = true_sleep_source_2;
      elseif true_states{trial}(2,t) == 3
          food = true_food_source_3;
           water = true_water_source_3;
           sleep = true_sleep_source_3;
      else
          food = true_food_source_4;
           water = true_water_source_4;
           sleep = true_sleep_source_4;
      end
      
 %   displayGridWorld(true_states{trial}(1,t),food,water,sleep, hill_1, 1, trial, t, time_since_resources, prefs{2}, states_history)
    g = {};
    
    y{2} = normalise_matrix(a{2});
    y{1} = A{1};
    y{3} = A{3};

    if horizon == 0
        horizon = 1;
    end
    temp_Q = Q;
    temp_Q{t,2} = temp_Q{t,2}';
    P = calculate_posterior(temp_Q,y,O,t);
    long_term_memory =0;
    a_complexity = 0;
    current_pos = find(cumsum(P{t,1})>=rand,1);
   
    optimal_traj = [];
    % Start tree search from current time point
    best_actions = [];
    
    global trajectory;
    trajectory = {};
    
    history = '';
    global tree_history;
    tree_history={};
    
    global searches;
    searches = 0;
    
    global post_calcs;
    post_calcs = 0;
    
    global auto_rest;
    auto_rest = 0;
    
    
    [G,Q, D, short_term_memory, long_term_memory, optimal_traj, best_actions, Tree] = tree_search_frwd(long_term_memory, short_term_memory, O, Q ,a, A,y, D, B,B, t, T, t+horizon, time_since_food, time_since_water, time_since_sleep, resource_locations, current_pos, true_t, chosen_action, a_complexity, surety, simulated_time, time_since_food, time_since_water, time_since_sleep, 0, optimal_traj, best_actions, history, k_factor, mct, num_mct);
    %TREE_SEARCH_HISTORY{trial,t} = tree_history;
    TREE_SEARCH_HISTORY{trial,t} = [nnz(short_term_memory), searches, numel(fieldnames(tree_history)), G, post_calcs];  
    short_term_memory(:,:,:,:,:) = 0; %reseting over and over
    chosen_action(t) = best_actions(1);
    t = t+1;
    % end loop over time points

end
survival(trial) = t;
action_history(trial,:) = chosen_action;
save(file_name, 't' );
save(matfilename, 'TREE_SEARCH_HISTORY' );

%save(matfilename, 'TREE_SEARCH_HISTORY');

t = 1;
time_since_food = 0;
time_since_water = 0;
time_since_sleep = 0;
end
%fclose(fid);
%fprintf(fid, '%f\n', 'targetted forgetting');

%% Auxilary Functions


% function a = displayGridWorld(agent_position, food_position_1,water_position_1,sleep_position_1,hill_1_pos,alive_status, trial, t, time_since_resources, prefs, states_history)
% if alive_status == 1
%     agent_text = 'A';
% else 
%     agent_text = 'Dead';
% end
% 
% agent_dim1 = 0;
% if agent_position <= 10
%     agent_dim2 = 1;
%     agent_dim1 = agent_position;
% elseif agent_position < 21
%     agent_dim2 = 2;
%     agent_dim1 = agent_position - 10;
% elseif agent_position < 31
%     agent_dim2 = 3;
%     agent_dim1 = agent_position - 20;
% elseif agent_position < 41
%     agent_dim2 = 4;
%     agent_dim1 = agent_position - 30;
% elseif agent_position < 51
%     agent_dim2 = 5;
%     agent_dim1 = agent_position - 40;
% elseif agent_position < 61
%     agent_dim2 = 6;
%     agent_dim1 = agent_position - 50;
% elseif agent_position < 71
%     agent_dim2 = 7;
%     agent_dim1 = agent_position - 60;
% elseif agent_position < 81
%     agent_dim2 = 8;
%     agent_dim1 = agent_position - 70;
% elseif agent_position < 91
%     agent_dim2 = 9;
%     agent_dim1 = agent_position - 80;
% else
%     agent_dim2 = 10;
%     agent_dim1 = agent_position - 90;
% end
% 
% locations_1 = [];
% hill_1_dim2 = idivide(int16(hill_1_pos),10,'floor')+1;
% hill_1_dim1 = rem(hill_1_pos,10);
% if hill_1_dim1 == 0
%     if hill_1_dim2 ~= 1
%         hill_1_dim2 = hill_1_dim2-1;
%     end
%     hill_1_dim1 = 10;
% end
% 
% food_1_dim2 = idivide(int16(food_position_1),10,'floor')+1;
% food_1_dim1 = rem(food_position_1,10);
% if food_1_dim1 == 0
%     if food_1_dim2 ~= 1
%         food_1_dim2 = food_1_dim2-1;
%     end
%     food_1_dim1 = 10;
% end
% locations_1(end+1) = food_1_dim1;
% 
% water_1_dim2 = idivide(int16(water_position_1), 10, 'floor')+1;
% water_1_dim1 = rem(water_position_1, 10);
% if water_1_dim1 == 0
%     water_1_dim1 = 10;
%     if water_1_dim2 ~= 1
%         water_1_dim2 = water_1_dim2-1;
%     end
% end
% 
% sleep_1_dim2 = idivide(int16(sleep_position_1), 10, 'floor')+1;
% sleep_1_dim1 = rem(sleep_position_1, 10);
% if sleep_1_dim1 == 0
%     sleep_1_dim1 = 10;
%     if sleep_1_dim2 ~= 1
%         sleep_1_dim2 = sleep_1_dim2-1;
%     end
% end
% locations_1(end+1) = sleep_1_dim1;
% 
% h1=figure(1);
% 
% set(h1,'name','gridworld');
% h1.Position = [9.0000   64.3333  665.3333  535.3333];
% [X,Y]=meshgrid(1:11,1:11);
% plot(Y,X,'k'); hold on; axis off
% plot(X,Y,'k');hold off; axis off
% hold off;
% I=(1);
% surface(I);
% h=linspace(0.5,1,64);
% q=1;
% x=linspace(1.5,10.5,10);
% y=linspace(1.5,10.5,10);
% 
% text(1, 11.5, ['trial = ' num2str(trial)], 'FontSize', 7, 'VerticalAlignment', 'bottom');
% text(3, 11.5, ['time step = ' num2str(t)], 'FontSize', 7, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
% 
% 
% text(7, 11.5, ['ts food = ' num2str(time_since_resources(1))], 'FontSize', 7, 'VerticalAlignment', 'bottom');
% text(8, 11.5, ['ts water = ' num2str(time_since_resources(2))], 'FontSize', 7, 'VerticalAlignment', 'bottom');
% text(9, 11.5, ['ts sleep = ' num2str(time_since_resources(3))], 'FontSize', 7, 'VerticalAlignment', 'bottom');
% text(8, 11.2, ['prefs = ' num2str(prefs)], 'FontSize', 7, 'VerticalAlignment', 'bottom');
% 
% for n=1:10
%     for p=1:10
%         if n == agent_dim1 & p == agent_dim2
%             text(y(n)-.2,x(p),agent_text,'FontSize',16);
%             q=q+1;
%         end
%         
%         if (n == food_1_dim1 & p == food_1_dim2) 
%             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
%             q=q+1;
%         end
%         
%         if (n == water_1_dim1 & p == water_1_dim2) 
%             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
%             q=q+1;
%         end
%         
%         
%         if (n == hill_1_dim1 & p == hill_1_dim2)
%             text(y(n)-.2,x(p)+.3,'Hill','FontSize',16, 'FontWeight','bold');
%             q=q+1;
%         end
%        
%         
%         if (n == sleep_1_dim1 & p == sleep_1_dim2)
%             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
%             q=q+1;
%         end
% 
%         
%     end
% end
% 
% % Create a new figure for the agent's trajectory
% % ... [rest of the function]
% 
% % Create a new figure for the agent's trajectory
% h2 = figure(2);
% h2.Position = [693.6667   83.6667  510.6667  474.6667];
% figure(h2);  % Bring h2 to the foreground
% clf(h2);     % Clear the figure content
% 
% title('Agent Trajectory');
% xlabel('Column');
% ylabel('Row');
% xlim([1, 10]);
% ylim([1, 10]);
% grid on;
% hold on;
% 
% % Extract the agent's trajectory for the current trial
% current_trajectory = states_history(trial, :);
% current_trajectory = current_trajectory(current_trajectory ~= 0); % Removing zero values
% 
% % Reverse the trajectory so we plot the oldest state first
% current_trajectory = fliplr(current_trajectory);
% 
% if length(current_trajectory) > 10
%     current_trajectory = current_trajectory(1:10);
% end
% 
% % Determine the color and size fading factors
% numStates = length(current_trajectory);
% colorFading = linspace(1, .1, numStates); % Starting from faint to solid
% sizeFading = linspace(20, 1, numStates);   % Starting from small to large
% 
% % Convert the trajectory states to (row, column) coordinates and plot them
% for idx = 1:numStates
%     state = current_trajectory(idx);
%     [row, col] = stateToCoordinates(state);
%     plot(col, row, 'o', 'Color', [0 0 1 colorFading(idx)], 'MarkerSize', sizeFading(idx));  % Plot using fading color and size
% end
% 
% hold off;
% 
% % ... [rest of the function]
% 
% % Helper function to convert a state to its corresponding (row, column) coordinates
% end


end




        
        
   



   
