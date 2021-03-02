%function to generate osie fixation maps using fixation data
% We are using antonioGaussian function from the saliency benchmarking code
% provided in https://github.com/cvzoya/saliency

function generate_osie_fixation_maps(osie_dataset_root)
    osie_stimuli_dir = sprintf('%s/data/stimuli/', osie_dataset_root);
    osie_fixmap_dir = sprintf('%s/data/fixation_maps/', osie_dataset_root);

    if ~exist(osie_fixmap_dir, 'dir')
        mkdir(osie_fixmap_dir);
    end

    fixations_mat = load(sprintf('%s/data/eye/fixations.mat', osie_dataset_root));
    fixations = fixations_mat.fixations;

    for i = 1:length(fixations)
        s_img = imread([osie_stimuli_dir fixations{i}.img]);

        f_img = zeros(size(s_img, 1), size(s_img, 2));

        for s = 1:length(fixations{i}.subjects)
            fix_x = round(fixations{i}.subjects{s}.fix_x);
            fix_y = round(fixations{i}.subjects{s}.fix_y);

            for j = 1:length(fix_x)
                f_img(fix_y(j), fix_x(j)) = 1;
            end

        end
        [f_img, ~] = antonioGaussian(f_img, 8);
    
        %uncomment to see the results
        %figure(1); clf; subplot(1, 2, 1); imagesc(s_img);
        %subplot(1, 2, 2); imagesc(f_img);
        %drawnow
        t = mat2gray(f_img);
        imwrite(mat2gray(f_img), sprintf('%s/%s', osie_fixmap_dir, fixations{i}.img));
    end

end
