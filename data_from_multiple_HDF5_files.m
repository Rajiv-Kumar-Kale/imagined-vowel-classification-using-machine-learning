function    out = data_from_multiple_HDF5_files( folder, glob )
    sad = dir( fullfile( folder, glob ) );
    len = length( sad );
    out = struct( 'name', [], 'subject', [], 'tr_vs_test', [], 'task', [],  'label',[], 'fdata',[] ,'delta',[],'theta',[],'alpha',[],'beta',[]);
    for jj = 1 : len
        out(jj).name = sad(jj).name;
        c = strsplit(out(jj).name,'_');
        d = char(c(3));
        d = strsplit(d,'.');
        d = char(d(1));
        d = [d(2) d(3)];
        out(jj).task = str2num(char(d));
        out(jj).tr_vs_test = char(c(2));
        k = char(c(1));
        out(jj).subject = str2num(char(k(2)));
        [out(jj).label, out(jj).fdata,out(jj).delta,out(jj).theta,out(jj).alpha,out(jj).beta ]= read_one_file( fullfile( folder, sad(jj).name ));
        
    end
end
