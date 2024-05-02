
%%%%%%%%%%%% code for graphical depiction of simulations %%%%%%%%%%%%%

function a = displayGridWorld(agent_position, food_position_1, water_position_1, sleep_position_1, hill_1_pos, alive_status)

    if alive_status == 1
        agent_text = 'A';
    else
        agent_text = 'Dead';
    end

    agent_dim1 = 0;

    if agent_position <= 10
        agent_dim2 = 1;
        agent_dim1 = agent_position;
    elseif agent_position < 21
        agent_dim2 = 2;
        agent_dim1 = agent_position - 10;
    elseif agent_position < 31
        agent_dim2 = 3;
        agent_dim1 = agent_position - 20;
    elseif agent_position < 41
        agent_dim2 = 4;
        agent_dim1 = agent_position - 30;
    elseif agent_position < 51
        agent_dim2 = 5;
        agent_dim1 = agent_position - 40;
    elseif agent_position < 61
        agent_dim2 = 6;
        agent_dim1 = agent_position - 50;
    elseif agent_position < 71
        agent_dim2 = 7;
        agent_dim1 = agent_position - 60;
    elseif agent_position < 81
        agent_dim2 = 8;
        agent_dim1 = agent_position - 70;
    elseif agent_position < 91
        agent_dim2 = 9;
        agent_dim1 = agent_position - 80;
    else
        agent_dim2 = 10;
        agent_dim1 = agent_position - 90;
    end

    locations_1 = [];
    hill_1_dim2 = idivide(int16(hill_1_pos), 10, 'floor') + 1;
    hill_1_dim1 = rem(hill_1_pos, 10);

    if hill_1_dim1 == 0

        if hill_1_dim2 ~= 1
            hill_1_dim2 = hill_1_dim2 - 1;
        end

        hill_1_dim1 = 10;
    end

    food_1_dim2 = idivide(int16(food_position_1), 10, 'floor') + 1;
    food_1_dim1 = rem(food_position_1, 10);

    if food_1_dim1 == 0

        if food_1_dim2 ~= 1
            food_1_dim2 = food_1_dim2 - 1;
        end

        food_1_dim1 = 10;
    end

    locations_1(end + 1) = food_1_dim1;
    % food_2_dim2 = idivide(int16(food_position_2), 10, 'floor')+1;
    % food_2_dim1 = rem(food_position_2,10);
    % if food_2_dim1 == 0
    %     food_2_dim1 = 10;
    %     if food_2_dim2 ~= 1
    %         food_2_dim2 = food_2_dim2-1;
    %     end
    % end
    % locations_1(end+1) = food_2_dim1;
    % food_3_dim2 = idivide(int16(food_position_3), 10, 'floor')+1;
    % food_3_dim1 = rem(food_position_3, 10);
    % if food_3_dim1 == 0
    %     food_3_dim1 = 10;
    %     if food_3_dim2 ~= 1
    %         food_3_dim2 = food_3_dim2-1;
    %     end
    % end
    % locations_1(end+1) = food_3_dim1;
    %
    % food_4_dim2 = idivide(int16(food_position_4), 10, 'floor')+1;
    % food_4_dim1 = rem(food_position_4, 10);
    % if food_4_dim1 == 0
    %     food_4_dim1 = 10;
    %     if food_4_dim2 ~= 1
    %         food_4_dim2 = food_4_dim2-1;
    %     end
    % end
    % locations_1(end+1) = food_4_dim1;

    water_1_dim2 = idivide(int16(water_position_1), 10, 'floor') + 1;
    water_1_dim1 = rem(water_position_1, 10);

    if water_1_dim1 == 0
        water_1_dim1 = 10;

        if water_1_dim2 ~= 1
            water_1_dim2 = water_1_dim2 - 1;
        end

    end

    % water_2_dim2 = idivide(int16(water_position_2), 10, 'floor')+1;
    % water_2_dim1 = rem(water_position_2, 10);
    % if water_2_dim1 == 0
    %     water_2_dim1 = 10;
    %     if water_2_dim2 ~= 1
    %         water_2_dim2 = water_2_dim2 - 1;
    %     end
    % end
    % locations_1(end+1) = water_2_dim1;
    %
    % water_3_dim2 = idivide(int16(water_position_3), 10, 'floor')+1;
    % water_3_dim1 = rem(water_position_3, 10);
    % if water_3_dim1 == 0
    %     water_3_dim1 = 10;
    %     if water_3_dim2 ~= 1
    %         water_3_dim2 = water_3_dim2 - 1;
    %     end
    % end
    % locations_1(end+1) = water_2_dim1;
    %
    % water_4_dim2 = idivide(int16(water_position_4), 10, 'floor')+1;
    % water_4_dim1 = rem(water_position_4, 10);
    % if water_4_dim1 == 0
    %     water_4_dim1 = 10;
    %     if water_4_dim2 ~= 1
    %         water_4_dim2 = water_4_dim2 - 1;
    %     end
    % end
    % locations_1(end+1) = water_2_dim1;
    sleep_1_dim2 = idivide(int16(sleep_position_1), 10, 'floor') + 1;
    sleep_1_dim1 = rem(sleep_position_1, 10);

    if sleep_1_dim1 == 0
        sleep_1_dim1 = 10;

        if sleep_1_dim2 ~= 1
            sleep_1_dim2 = sleep_1_dim2 - 1;
        end

    end

    locations_1(end + 1) = sleep_1_dim1;
    % sleep_2_dim2 = idivide(int16(sleep_position_2), 10, 'floor')+1;
    % sleep_2_dim1 = rem(sleep_position_2, 10);
    % if sleep_2_dim1 == 0
    %     sleep_2_dim1 = 10;
    %     if sleep_2_dim2 ~= 1
    %         sleep_2_dim2 = sleep_2_dim2-1;
    %     end
    % end
    %
    % sleep_3_dim2 = idivide(int16(sleep_position_3), 10, 'floor')+1;
    % sleep_3_dim1 = rem(sleep_position_2, 10);
    % if sleep_3_dim1 == 0
    %     sleep_3_dim1 = 10;
    %     if sleep_3_dim2 ~= 1
    %         sleep_3_dim2 = sleep_3_dim2-1;
    %     end
    % end
    %
    % sleep_4_dim2 = idivide(int16(sleep_position_4), 10, 'floor')+1;
    % sleep_4_dim1 = rem(sleep_position_4, 10);
    % if sleep_4_dim1 == 0
    %     sleep_4_dim1 = 10;
    %     if sleep_4_dim2 ~= 1
    %         sleep_4_dim2 = sleep_4_dim2-1;
    %     end

    h1 = figure(1);
    set(h1, 'name', 'gridworld');
    h1.Position = [400 200 800 700];
    [X, Y] = meshgrid(1:11, 1:11);
    plot(Y, X, 'k'); hold on; axis off
    plot(X, Y, 'k'); hold off; axis off
    hold off;
    I = (1);
    surface(I);
    h = linspace(0.5, 1, 64);
    %h=[h',h',h'];
    %set(gcf,'Colormap',h);
    q = 1;
    x = linspace(1.5, 10.5, 10);
    y = linspace(1.5, 10.5, 10);
    %empty_pref =sprintf('%.3f',preference_values(1));
    %food_pref =sprintf('%.3f',preference_values(2));
    %water_pref =sprintf('%.3f',preference_values(3));
    %sleep_pref =sprintf('%.3f',preference_values(4));
    for n = 1:10

        for p = 1:10

            if n == agent_dim1 & p == agent_dim2
                text(y(n) - .2, x(p), agent_text, 'FontSize', 16);
                q = q + 1;

            end

            if (n == food_1_dim1 & p == food_1_dim2)
                text(y(n) - .2, x(p) + .3, 'F', 'FontSize', 16, 'FontWeight', 'bold');
                %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
                q = q + 1;
            end

            %          if (n == food_2_dim1 & p == food_2_dim2)
            %             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
            %             q=q+1;
            %          end
            %
            %           if (n == food_3_dim1 & p == food_3_dim2)
            %             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
            %             q=q+1;
            %           end
            %
            %            if (n == food_4_dim1 & p == food_4_dim2)
            %             text(y(n)-.2,x(p)+.3,'F','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,food_pref,'FontSize', 12);
            %             q=q+1;
            %         end

            if (n == water_1_dim1 & p == water_1_dim2)
                text(y(n) - .2, x(p) + .3, 'W', 'FontSize', 16, 'FontWeight', 'bold');
                %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
                q = q + 1;
            end

            %          if (n == water_2_dim1 & p == water_2_dim2)
            %             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
            %             q=q+1;
            %          end
            %
            %           if (n == water_3_dim1 & p == water_3_dim2)
            %             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
            %             q=q+1;
            %           end
            %
            %            if (n == water_4_dim1 & p == water_4_dim2)
            %             text(y(n)-.2,x(p)+.3,'W','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
            %             q=q+1;
            %         end

            if (n == hill_1_dim1 & p == hill_1_dim2)
                text(y(n) - .2, x(p) + .3, 'Hill', 'FontSize', 16, 'FontWeight', 'bold');
                %text(y(n)-.2,x(p)-.3,water_pref,'FontSize', 12);
                q = q + 1;
            end

            if (n == sleep_1_dim1 & p == sleep_1_dim2)
                text(y(n) - .2, x(p) + .3, 'S', 'FontSize', 16, 'FontWeight', 'bold');
                %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
                q = q + 1;
            end

            %         if (n == sleep_3_dim1 & p == sleep_3_dim2)
            %             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
            %             q=q+1;
            %         end
            %
            %         if (n == sleep_3_dim1 & p == sleep_3_dim2)
            %             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
            %             q=q+1;
            %         end
            %
            %         if (n == sleep_4_dim1 & p == sleep_4_dim2)
            %             text(y(n)-.2,x(p)+.3,'S','FontSize',16, 'FontWeight','bold');
            %             %text(y(n)-.2,x(p)-.3,sleep_pref,'FontSize', 12);
            %             q=q+1;
            %         end

            %if ~(n == sleep_dim1 && p == sleep_dim2) && ~(n == food_1_dim1 && p == food_1_dim2) && ~(n == food_2_dim1 && p == food_2_dim2) && ~(n == food_3_dim1 && p == food_3_dim2) && ~(n == water_1_dim1 && p == water_1_dim2) && ~(n ==water_2_dim1 && p == water_2_dim2)
            %  text(y(n)-.2,x(p)-.3,empty_pref,'FontSize', 12);
            %end

        end

    end

    %pause(0.5)

end

%--------------------------------------------------------------------------
