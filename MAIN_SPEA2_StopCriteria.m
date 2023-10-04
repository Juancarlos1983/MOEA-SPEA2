% =====================================================================  
% --------------------------------------------------------------------- 
% ---------------------   JUAN CARLOS TICONA  -------------------------
% ---------- INSTITUTO DE PESQUISAS HIDRAULICAS (IPH) UFRGS  ----------
% -------------------------- OUTUBRO DE 2023 --------------------------    
% --------------------------------------------------------------------- 
% =====================================================================
% 
t = zeros(3,10);
for arquivo = 1:10
    save ('salva.mat','arquivo','t');    
    clc;
    clear;
    close all;
    load ('salva.mat');
    disp(arquivo);
    tic;
    primeira = 0;
    segunda = 0;
    Count = 0;
    Count_Max = 10;
    F1min = 0.;
    F2min = 0;
    d = [];
    numPop = [];
    GD0=10000;
 
    %% Definição do Problema
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Valores utilizados no HyMOD_5p
%     nVar = 5;    %Variáveis de decisão.
%     % Parâmetros do Tank-Model a serem calibrados.
%     % X = [Smax, b, a, kf, ks]
%     %Limite inferior de cada uma das variáveis de decisao:
%     VarMin=[10.0,0.0,0.0,0.15,0.0];   
%     %Limite superior de cada uma das variáveis de decisao:
%     VarMax=[2000,7.0,1.0,1.0,0.15];    

%     VarSize = [1 nVar];         %Size of Decision Variables Matrix
%     FO = @(x) FO_HYMOD(x);       % Function Objective    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Valores utilizados no GR4J_4p
%     nVar = 6;    %Variáveis de decisão.
%     % Parâmetros do Tank-Model a serem calibrados.
%     % X = [Smax, kf, Rmax, T, So, Fo]
%     %Limite inferior de cada uma das variáveis de decisao:
%     VarMin=[0.01,-10.0,10.0,0.5,1,1]; 
%     %Limite superior de cada uma das variáveis de decisao:
%     VarMax=[1500,5.0,500.0,4.0,100,200];  

%     VarSize = [1 nVar];         %Size of Decision Variables Matrix
%     FO = @(x) FO_GR4J(x);       % Function Objective   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %Valores utilizados no GR5J_7p nVar = 7;     
%     nVar = 7;    %Variáveis de decisão.
%     % Parâmetros do Tank-Model a serem calibrados.
%     % X = [Smax, kf, Rmax, T, K, So, Fo]
%     %Limite inferior de cada uma das variáveis de decisao:
%     VarMin=[0.01,-10.0,1.0,0.5,0.001,1,1]; 
%     %Limite superior de cada uma das variáveis de decisao:
%     VarMax=[2000,5.0,500.0,4.0,1.0,100,200];   

%     VarSize = [1 nVar];         %Size of Decision Variables Matrix
%     FO = @(x) FO_GR5J(x);       % Function Objective   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %Valores utilizados no IPH2_7p
%     nVar = 7;    %Variáveis de decisão.
%     % Parâmetros do Tank-Model a serem calibrados.
%     % X = [Io, Ib, h, Ksup, Ksub, Rmax, a]
%     %Limite inferior de cada uma das variáveis de decisao:
%     VarMin = [10.0,0.1,0.01,0.01,10.0,0.0,0.01];  
%     %Limite superior de cada uma das variáveis de decisao:
%     VarMax = [300.0,10.0,0.99,10.0,500.0,9.0,20.0];

%     VarSize = [1 nVar];         %Size of Decision Variables Matrix
%     FO = @(x) FO_IPH2(x);       % Function Objective  
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %Valores utilizados no Tank-Model 3 hidro
%     nVar = 9;    %Variáveis de decisão.
%     % Parâmetros do Tank-Model a serem calibrados.
%     % X = [H1, H2, H3, a1, a2, a3, a4, b1, b2]
%     %Limite inferior de cada uma das variáveis de decisao:
%     VarMin=[10,10,10,0.09,0.09,0.09,0.01,0.01,0.01];   
%     %Limite superior de cada uma das variáveis de decisao:
%     VarMax=[70,45,70,0.5,0.5,0.5,0.1,0.1,0.1];    

%     VarSize = [1 nVar];         %Size of Decision Variables Matrix
%     FO = @(x) FO_TANK3hidro(x);      % Function Objective   
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Valores utilizados no Tank-Model 4 hidro
    nVar = 16;    %Variáveis de decisão. 
    % Parâmetros do Tank-Model a serem calibrados.
    % X = [HI1, HI2, HI3, HI4, HA1, HA2, HB1, HC1, a1, a2, b1, c1, d1, a0, b0, c0]
    %Limite inferior de cada uma das variáveis de decisão:
    VarMin=[5,10,50,100,10,10,10,10,0.09,0.09,0.09,0.01,0.001,0.01,0.01,0.001];   
    %Limite superior de cada uma das variáveis de decisão:
    VarMax=[75,70,200,500,70,45,70,70,0.5,0.5,0.5,0.1,0.01,0.1,0.1,0.1];  

    VarSize = [1 nVar];          % Size of Decision Variables Matrix
    FO = @(x) FO_TANK4hidro(x);  % Function Objective

    % Número de funções objetivas
    nObj = numel(FO(VarMin + (VarMax - VarMin)*rand));
        
    %% SPEA2 Parâmetros
    
    MaxIt=300;         % Número Máximo de Iterações
    nPop=50;           % Tamanho da população

    nArchive=nPop;        % Tamanho do arquivo

    K=round(sqrt(nPop+nArchive));  % KNN Parâmetro

    pCrossover=0.7;     % Porcentagem de soluções obtidas do Cruzamento
    nCrossover=round(pCrossover*nPop/2)*2;

    nMutation=nPop-nCrossover;

    crossover_params.gamma=0.01;
    crossover_params.VarMin=VarMin;
    crossover_params.VarMax=VarMax;

    mutation_params.h=0.1;
    mutation_params.VarMin=VarMin;
    mutation_params.VarMax=VarMax;

    StopCrit = zeros(MaxIt,2); %variaveris do criterio de parada
    
    %% Inicialização
    
    disp('Staring SPEA2 ...');
    
    empty_individual.Position=[];
    empty_individual.Cost=[];
    empty_individual.S=[];
    empty_individual.R=[];
    empty_individual.sigma=[];
    empty_individual.sigmaK=[];
    empty_individual.D=[];
    empty_individual.F=[];

    pop=repmat(empty_individual,nPop,1);
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Criação das populações para os testes
%     for i=1:nPop
%         pop(i).Position=VarMin + (VarMax - VarMin)*rand;
%         pop(i).Cost=FO(pop(i).Position);
%     end
    
%     save(sprintf('população_ijui_TANK4_%d.mat',arquivo),'pop')
    
    %%
    population = load (sprintf('população_ijui_TANK4_%d.mat',arquivo));
    for i=1:nPop
        pop(i).Position=population.pop(i).Position;
        pop(i).Cost=population.pop(i).Cost;
    end
    
    archive=[];

    %% NSGA-III Loop principal
    
    for it=1:MaxIt
        if it == 1
            Q=[pop
            archive];

            nQ=numel(Q);

            dom=false(nQ,nQ);

            for i=1:nQ
                Q(i).S=0;
            end

            for i=1:nQ
                for j=i+1:nQ

                    if Dominates(Q(i),Q(j))
                        Q(i).S=Q(i).S+1;
                        dom(i,j)=true;

                    elseif Dominates(Q(j),Q(i))
                        Q(j).S=Q(j).S+1;
                        dom(j,i)=true;

                    end

                end
            end

            S=[Q.S];
            for i=1:nQ
                Q(i).R=sum(S(dom(:,i)));
            end

            Z=[Q.Cost]';
            SIGMA=pdist2(Z,Z,'seuclidean');
            SIGMA=sort(SIGMA);
            for i=1:nQ
                Q(i).sigma=SIGMA(:,i);
                Q(i).sigmaK=Q(i).sigma(K);
                Q(i).D=1/(Q(i).sigmaK+2);
                Q(i).F=Q(i).R+Q(i).D;
            end

            nND=sum([Q.R]==0);
            if nND<=nArchive
                F=[Q.F];
                [F, SO]=sort(F);
                Q=Q(SO);
                archive=Q(1:min(nArchive,nQ));        
            else
                SIGMA=SIGMA(:,[Q.R]==0);
                archive=Q([Q.R]==0);

                k=2;
                while numel(archive)>nArchive
                    while min(SIGMA(k,:))==max(SIGMA(k,:)) && k<size(SIGMA,1)
                        k=k+1;
                    end

                    [~, j]=min(SIGMA(k,:));

                    archive(j)=[];
                    SIGMA(:,j)=[];
                end        
            end
        end
            
        % Crossover
        popc=repmat(empty_individual,nCrossover/2,2);
        for c=1:nCrossover/2

            p1=BinaryTournamentSelection(archive,[archive.F]);
            p2=BinaryTournamentSelection(archive,[archive.F]);

            [popc(c,1).Position, popc(c,2).Position]=Crossover(p1.Position,p2.Position,crossover_params);

            popc(c,1).Cost=FO(popc(c,1).Position);
            popc(c,2).Cost=FO(popc(c,2).Position);

        end
        popc=popc(:);

        % Mutação
        popm=repmat(empty_individual,nMutation,1);
        for m=1:nMutation

            p=BinaryTournamentSelection(archive,[archive.F]);

            popm(m).Position=Mutate(p.Position,mutation_params);

            popm(m).Cost=FO(popm(m).Position);

        end

        % Create New Population
        pop=[popc
             popm];      
    
         
        % Ordenar População e execute a seleção
        Q=[pop
            archive];

            nQ=numel(Q);

            dom=false(nQ,nQ);

        for i=1:nQ
            Q(i).S=0;
        end

        for i=1:nQ
            for j=i+1:nQ

                if Dominates(Q(i),Q(j))
                    Q(i).S=Q(i).S+1;
                    dom(i,j)=true;

                elseif Dominates(Q(j),Q(i))
                    Q(j).S=Q(j).S+1;
                    dom(j,i)=true;

                end

            end
        end

        S=[Q.S];
        for i=1:nQ
            Q(i).R=sum(S(dom(:,i)));
        end

        Z=[Q.Cost]';
        SIGMA=pdist2(Z,Z,'seuclidean');
        SIGMA=sort(SIGMA);
        for i=1:nQ
            Q(i).sigma=SIGMA(:,i);
            Q(i).sigmaK=Q(i).sigma(K);
            Q(i).D=1/(Q(i).sigmaK+2);
            Q(i).F=Q(i).R+Q(i).D;
        end

        nND=sum([Q.R]==0);
        if nND<=nArchive
            F=[Q.F];
            [F, SO]=sort(F);
            Q=Q(SO);
            archive=Q(1:min(nArchive,nQ));        
        else
            SIGMA=SIGMA(:,[Q.R]==0);
            archive=Q([Q.R]==0);

            k=2;
            while numel(archive)>nArchive
                while min(SIGMA(k,:))==max(SIGMA(k,:)) && k<size(SIGMA,1)
                    k=k+1;
                end

                [~, j]=min(SIGMA(k,:));

                archive(j)=[];
                SIGMA(:,j)=[];
            end        
        end

        pop=archive([archive.R]==0);% Soluções não dominadas do Pareto Front

        % Calculo de Criterios de Parada[nF GD] = StopingCriteria(pop);
        StopCrit(it,:) = [nF GD];
        
        % Criterio de parada das iterações
        Melhores=zeros(nF,nVar+3);
        for i = 1:nF
            Melhores(i,:) = [pop(i).Position 1-pop(i,1).Cost(1) 1-pop(i,1).Cost(2) sqrt((pop(i,1).Cost(1)-F1min)^2 + (pop(i,1).Cost(2)-F2min)^2)];
%             Melhores(i,:) = [pop(i).Position pop(i,1).Cost(1) pop(i,1).Cost(2) sqrt((pop(i,1).Cost(1)-F1min)^2 + (pop(i,1).Cost(2)-F2min)^2)];
        end 
        if nF > nPop-1 && sum(isnan(Melhores(:,nVar + 1))) == 0 && sum(isnan(Melhores(:,nVar + 2))) == 0 && it > 125
            Count = Count+1;
        else
            Count=0;
        end
        %% Results
        if Count > Count_Max && primeira == 0
            disp('pparou CP1');
            % Results              
            Melhores1 = sortrows(Melhores, nVar + 3);
            xlswrite(sprintf('pareto ijui SPEA2_3CP+TANK4_T2.xlsx'),Melhores1,num2str(['CP1_' num2str([arquivo it])]))
            primeira = 1;
            %break;
            t(1,arquivo) = toc;
        end
        %% Criterio padrao CP2
        if GD < 0.15
            if abs(GD0 - GD) < 0.001 && segunda == 0;
            disp('parou CP2');            
            %% Results
            Melhores2 = sortrows(Melhores, nVar + 3);
            xlswrite(sprintf('pareto ijui SPEA2_3CP+TANK4_T2.xlsx'),Melhores2,num2str(['CP2_' num2str([arquivo it])]))
            segunda = 1;
            %break;
            t(2,arquivo) = toc;
            else
            GD0 = GD;
            end
        end
        %% Criterio padrao
        if it == MaxIt
            %% Results
            disp('parou CP3');
            disp('Optimization Terminated.');
            Melhores3 = sortrows(Melhores, nVar + 3);
            xlswrite(sprintf('pareto ijui SPEA2_3CP+TANK4_T2.xlsx'),Melhores3,num2str(['CP3_' num2str([arquivo it])]))
            xlswrite(sprintf('Medidas ijui SPEA2+TANK4_T2.xlsx'),StopCrit,num2str([arquivo it]),'A2')
            t(3,arquivo) = toc;
            break;
        end
    end
end
save ('tempo 3C_ijui_SPEA2_TANK4_T2.mat','t');