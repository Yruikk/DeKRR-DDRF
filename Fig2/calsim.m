function sim = calsim(vec_1,vec_2)
sim = (vec_1'*vec_2)/(norm(vec_1,2)*norm(vec_2,2));
end