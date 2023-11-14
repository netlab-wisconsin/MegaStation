% Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
%
% NVIDIA CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from NVIDIA CORPORATION is strictly prohibited.

function differences = struct_compare(s1, s2)
    differences = do_struct_compare(0, s1, s2);
end

function differences = do_struct_compare(diff_previous, s1, s2)
    differences = diff_previous;
    fnames = unique([ fieldnames(s1); fieldnames(s2) ]);
    for idx = 1:length(fnames)
        fname = fnames{idx};
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Check that the field exists in structure 1
        if ~isfield(s1, fname)
            fprintf('Field %s is not present in structure 1\n', fname);
            differences = differences + 1;
            continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Check that the field exists in structure 2            
        if ~isfield(s2, fname)
            fprintf('Field %s is not present in structure 2\n', fname);
            differences = differences + 1;
            continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Compare field types
        field1 = getfield(s1, fname);
        field2 = getfield(s2, fname);
        if ~strcmp(class(field1), class(field2))
            fprintf('Field %s is of class %s in struct 1 and %s in struct 2', ...
                    fname,                                                    ...
                    class(field1),                                            ...
                    class(field2));
            differences = differences + 1;
            continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Compare field dimensions
        if any(size(field1) ~= size(field2))
              fprintf('Size of field %s differs in struct 1 and struct 2: \n', fname);
              fprintf('\tstruct 1: ');
              fprintf('%d ', size(field1));
              fprintf('\n\tstruct 2: ');
              fprintf('%d ', size(field2));
              fprintf('\n');
              differences = differences + 1;
              continue
        end
        %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Compare individual array values
        if strcmp('struct', class(field1))
            for elem_idx = 1:length(field1)
                differences = do_struct_compare(differences, field1, field2);
            end
        else
            if any(field1 ~= field2, 'all')
                fprintf('Values in field %s differ\n', fname);
                differences = differences + 1;
            end
        end
    end % for idx = 1:length(fnames)
end
