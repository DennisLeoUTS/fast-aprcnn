%function res = cub_eval(output_dir, rm_res)

output_dir = '/home/zxu/workspace/fast-rcnn-test/output/svm/cub_test/vgg16_fast_rcnn_iter_40000_svm/';
file = 'head_detection_results.txt';
partid = 2;

f = fopen([output_dir file]);
C = textscan(f,'%s%f%f%f%f%f');
fclose(f);

[imn, m, n] = unique(C{1}, 'stable');
Nim = length(m);
os = zeros(1,length(m));
osp5 = zeros(1,length(m));
partexist = ones(1,length(m));

for i=1:length(m)
    if mod(i, 1000)==0
        fprintf('%d/%d: mean overlap>0.5: %.4f\n', i, length(m), mean(osp5(1:i)));
    end
    %{
    imname = imdb_test.image_at(i);
    imname = imname(40:end);
    tmp = cellfun(@(x) strcmp(x, imname), imn);
    tmp = find(tmp);
    if isempty(tmp)
        continue
    end
    inds = find(n==tmp);
    %}
    
    inds = find(n==i);
    tmp = inds(1);
    score = C{2}(tmp);
    n1 = C{1}(tmp);
    n2 = imdb_test.image_at(i);
    assert(strcmp(n2(40:end),n1))
    
    if score == -1
        continue
    end
    x1 = C{3}(tmp);
    y1 = C{4}(tmp);
    x2 = C{5}(tmp);
    y2 = C{6}(tmp);
    
    roisi = roidb_test.rois(i);
    clind = find(roisi.class==partid);
    
    if isempty(clind)
        partexist(i) = 0;
        continue
    end
    
    gt_box = roisi.boxes(clind,:);
    o = boxoverlap(gt_box, [x1,y1,x2,y2]);
    os(i) = o;
    osp5(i) = o>.5;
    
    %{
    fprintf('%d: %d\n', i, o>.5); 
    im = imread(imdb_test.image_at(i));
    clf
    imshow(im);
    hold on
    rectangle('Position',[x1,y1,x2-x1,y2-y1], 'EdgeColor', 'r', 'LineWidth', 5);
    rectangle('Position',[gt_box(1),gt_box(2),gt_box(3)-gt_box(1),gt_box(4)-gt_box(2)], 'EdgeColor', 'g', 'LineWidth', 5);
    pause;
    %}
end

mean(os(find(partexist)))
mean(osp5(find(partexist)))
%end

