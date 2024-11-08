function mu = michaelisMentenModel(params, N)
    muMax = params(1); % 최대 성장률 μmax
    KGR_NO3 = params(2); % NO3 농도 KGR-NO3
    mu = muMax .* N ./ (KGR_NO3 + N); % Michaelis-Menten 식
end
