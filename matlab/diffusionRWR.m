function Q = diffusionRWR(A, maxiter, restartProb)
	n = size(A, 1); % n表示矩阵A的行数

	% Add self-edge to isolated nodes
	A = A + diag(sum(A) == 0); 

	% Normalize the adjacency matrix 
	renorm = @(M) bsxfun(@rdivide, M, sum(M));
	P = renorm(A); 

	% Personalized PageRank
	restart = eye(n);
	Q = eye(n); 
    
	for i = 1 : maxiter
		Q_new = (1 - restartProb) * P * Q + restartProb * restart;  
		delta = norm(Q - Q_new, 'fro');
		Q = Q_new; 
		if delta < 1e-6		
			break;
        end
    end   
end
