# PGMs in Stan language

pg1 ="""
    data{
        int<lower=1> N; // number of peer grades observed
        int<lower=1> uu[N]; // index for gradee of each observed grade
        int<lower=1> vv[N]; // index for grader of each observed grade
        int<lower=0> Ns_obs;
        int<lower=0> Ns_miss;
        int ii_obs_p[Ns_obs];
        int ii_miss_p[Ns_miss];

        real z[N] ; // peer grades vector 
        real s_obs[Ns_obs]; // observed grades
    }

    transformed data{
        int<lower=1> S; // number of students
        S = Ns_obs + Ns_miss;

    }

    parameters{
        real s_miss[Ns_miss];
        real<lower=0> sigma;
        real b[S]; 
    }

    transformed parameters{
        real s[S];
        if (Ns_obs > 0)
          s[ii_obs_p] = s_obs;
        s[ii_miss_p] = s_miss;

    }

    model {
        sigma ~ inv_gamma(0.01, 0.01);
        b ~ normal(0, 1.);
        for (j in 1:S){
            s[j] ~ normal(5., 10.); // should be (5,10)      
        }
        for (n in 1:N)
          z[n] ~ normal(s[uu[n]] + b[vv[n]], sigma); 
    }
    generated quantities{
        real z_tilde[N];
        real T[S];
        for (n in 1:N)
            z_tilde[n] = normal_rng(s[uu[n]] + b[vv[n]], sigma); // truncation not accepted for _rng functions in Stan
        for (i in 1:S)
            T[i] = normal_rng(b[i], sigma);
    }
    """


bivariate = """

data{
        int<lower=1> N;
        int<lower=0> N_pairs; // number of peer-pairs having some assessment in common
        int<lower=0> Ns_miss;
        int<lower=0> Ns_obs;
        int N_ragged_z;
        vector[2] ragged_z[N_ragged_z]; 
        int lengths[N_pairs];
        int ii_obs_p[Ns_obs]; // location in s of seen ground truths
        int ii_miss_p[Ns_miss]; // location in s of unseen ground truths
        int gg1[N_pairs];
        int gg2[N_pairs];
        real<lower=0>  s_obs[Ns_obs]; // observed grades;
        int<lower=1> uu[N]; // index for gradee of each observed grade
        int<lower=1> vv[N]; // index for grader of each observed grade
        //real<lower=0> z[N];  // peer grades (uncomment)
        real z[N]; // no lo hago exclusivamente negativo
    }

    transformed data{
        int<lower=1> S = Ns_miss + Ns_obs;
    }

    parameters{
        real<lower=0> s_miss_p[Ns_miss];
        real<lower=0> mu_s; // teacher's mean
        real<lower=0> sigma_s; // teacher's variance
        vector<lower=0>[S] mu; // peers' means
        vector<lower=0>[S] sigma; // peers' variances
        cholesky_factor_corr[2] Lcorr[N_pairs];
        cholesky_factor_corr[2] Lcorr_s[S];
    }

    transformed parameters{

        vector<lower=0>[2] mu_vector[N_pairs]; // array of vectors of means
        vector<lower=0>[2] sigma_vector[N_pairs]; // array of vectors of variances
        vector<lower=0>[2] mu_vector_s[S];
        vector<lower=0>[2] sigma_vector_s[S];
        real<lower=0> s[S];

        if (Ns_obs > 0)
            s[ii_obs_p] = s_obs;
        s[ii_miss_p] = s_miss_p;

        for (i in 1:N_pairs) {
            mu_vector[i] = [ mu[gg1[i]], mu[gg2[i]] ]';
            sigma_vector[i] =  [ sigma[gg1[i]], sigma[gg2[i]] ]';
        }

        for (i in 1:S){
            mu_vector_s[i] = [ mu_s, mu[i] ]';
            sigma_vector_s[i] =  [ sigma_s, sigma[i] ]';        
        }
    }

    model { 
        int pos = 1;
        // very weakly-informative priors
        sigma_s ~ cauchy(0,25.); ////inv_gamma(0.01, 0.01);
        sigma ~ cauchy(0,25); //inv_gamma(0.01, 0.01);//
        mu_s ~ normal(5.5, 5);
        mu ~   normal(5.5, 5);

        //s ~ normal(mu_s, sigma_s);

        //for (i in 1:N){
        //    z[i] ~ normal(mu[vv[i]], sigma[vv[i]]);
        //}

        for (i in 1:N_pairs){
            Lcorr[i] ~ lkj_corr_cholesky(1.);
        }

        for (i in 1:S){
            Lcorr_s[i] ~ lkj_corr_cholesky(1.);

        }

        for (i in 1:N_pairs){
            segment(ragged_z, pos, lengths[i]) ~ multi_normal_cholesky(mu_vector[i], diag_pre_multiply(sigma_vector[i], Lcorr[i]));
            pos += lengths[i];
        }

        for (n in 1:N){
            vector[2] sz = [s[uu[n]],z[n]]';
            sz ~ multi_normal_cholesky(mu_vector_s[vv[n]], diag_pre_multiply(sigma_vector_s[vv[n]], Lcorr_s[vv[n]]));
        }
    }

    generated quantities{
        real z_tilde[N];
        for (i in 1:N){
            z_tilde[i] = normal_rng(mu[vv[i]], sigma[vv[i]]);
        }
    }

"""
