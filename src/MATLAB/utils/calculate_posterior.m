function P = calculate_posterior(P, A, O, t)

    if size(P{t, 2}, 2) > 1
        P{t, 2} = P{t, 2}';
    end

    for fact = 2:2
        L = 1;
        num = numel(A);

        for modal = 2:num
            obs = find(cumsum(O{modal, t}) >= rand, 1);
            temp = A{modal}(obs, :, :);
            temp = permute(temp, [3, 2, 1]);
            L = L .* temp;
        end

        %L = permute(L,[3,2,1]);
        for f = 1:2

            if f ~= fact

                if f == 2
                    LL = P{t, f} * L;
                else
                    LL = L * P{t, f}';
                end

            end

        end

        y = LL .* P{t, fact};
        P{t, fact} = normalise(y)';
    end

end
% function P = calculate_posterior(P, A, O, t)
%     % fprintf('Original size of P{t, 2}: [%d, %d]\n', size(P{t, 2}, 1), size(P{t, 2}, 2));

%     % Ensure P{t, 2} is a column vector or appropriately dimensioned
%     if size(P{t, 2}, 2) > 1
%         P{t, 2} = P{t, 2}';
%     end

%     % Compute L based on modalities
%     fact = 2;
%     L = 1;
%     num = numel(A);
%     for modal = 2:num
%         obs = find(cumsum(O{modal, t}) >= rand, 1);
%         temp = A{modal}(obs, :, :);
%         temp = permute(temp, [3, 2, 1]);
%         L = L .* temp;
%     end

%     % fprintf('Size of L before multiplication: [%d, %d]\n', size(L, 1), size(L, 2));
%     L = L';  % Ensure L is transposed if necessary

%     % Adjust LL calculation to be more transparent and correct
%     for f = 1:2
%         if f ~= fact
%             % fprintf('Multiplying P{t, %d} of size [%d, %d] with L of size [%d, %d]\n', f, size(P{t, f}, 1), size(P{t, f}, 2), size(L, 1), size(L, 2));
%             LL = P{t, f} * L;  % Ensure dimensions match
%         end
%     end

%     y = LL .* P{t, fact};
%     P{t, fact} = normalise(y)';

%     % fprintf('Updated size of P{t, %d}: [%d, %d]\n', fact, size(P{t, fact}, 1), size(P{t, fact}, 2));
% end
