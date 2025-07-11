function cell2csv(filename, cellArray)
    fid = fopen(filename, 'w');
    [rows, cols] = size(cellArray);
    for i = 1:rows
        for j = 1:cols
            var = cellArray{i,j};
            if isnumeric(var)
                fprintf(fid, '%.4f', var);
            else
                fprintf(fid, '%s', var);
            end
            if j < cols
                fprintf(fid, ',');
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
