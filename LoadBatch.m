function [X, Y, y] = LoadBatch(filename)
    dict = load(filename);
    X = double(dict.data');
    y = dict.labels + 1;
    Y = (y == 1:10)';
end