# EHU-Weka-Proiektua

##Deskribapena
3.mailako Erabakiak Hartzeko Euskarri Sistemak irakasgaian burututako proiektua. Testu meatzaritzako sailkapen gainbegiratuko (train eta dev) eta ez gainbegiratu (test) atazen inguruan zenbait funtzionalitate jorratzean datza.
Horretarako, java eta weka-ko liburutegiak erabili ditugu. Gure kasuan, paziente askoren hainbat datu-mediku emanda (id, moduloa, adina, sexua, lekua, azalpen-testua, klasea), hauek hilko diren gaixotasunaz iragarriko du. 


##Parte-hartzaileak
- Liam Suárez
- María Briones
- Ainhoa Sánchez


##Nola exekutatu
Proiektuak ondo funtzionatzeko, lehenik eta behin entregan datozen hiru liburutegiak deskargatu beharko dira:
commons-lang3-3.12.0.jar
weka.jar
opencsv-5.10.jar

Proiektu honetan, funtzionalitate bezain beste exekutagarri daude. Gainera, orden jakin bat jarraitu behar da. Honako hau izanda:

1. Aurreprozesamendua: Datuak .txt-ik .arff-ra bilakatu.

   1. getARRF.jar: Emandako datuetatik, train eta test multzoen .arff formatua lortzen da, biak batera.
    ```bash
      java - jar getARFF.jar train.csv test.csv train.arff test.arff
     ```
    
   2. getSplit.jar: Entrenamendu datuak train eta dev datu-sortetan banatu ditugu: train (%80) eta dev (%20). Train dataset-a entrenamendurako erabiliko dugu, eta berriz, dev dataset-a garapenerako erabiliko dugu.
    ```bash
      java -jar getSplit.jar train.arff train_split.arff test_split.arff 
     ```

   3. arff2bow.jar: .arff formatutik Bag Of Words (BOW) formatura pasatzeko balio duen programa egin dugu.
      ```bash
        java -jar arff2bow.jar train_split_RAW.arff dev_split_RAW.arff test_RAW.arff dictionary.txt train_split_BOW.arff dev_split_BOW.arff test_BOW.arff 
      ```
      
   4. fssInfoGain.jar: Atributu hautapena egingo da, atributu erredundanteak edo garrantzi gutxikoak kenduz. Hauek informazio irabaziaren arabera aukeratuko dira.
     ```bash
       java -jar fssInfoGain.jar train_split_BOW.arff dev_split_BOW.arff test_BOW.arff train_split_BOW_FSS.arff dev_split_BOW_FSS.arff test_BOW_FSS.arff  
     ```


2. Eredu optimoa eta itxarondako kalitatearen estimazioa:
   
   1. getBaseline.jar: Logistic Regression teknika erabiltzen da programa honetan.
     ```bash
       java -jar getBaseline.jar train_split_BOW_FSS.arff dev_split_BOW_FSS.arff test_BOW_FSS.arff ebaluazioa_baseline.txt iragarpenak_baseline.txt 
     ```
   
   2. parametroEkorketa.jar: Support Vector Machine teknika erabiltzen da programa honetan. Modelorako parametro hoberenen hautaketa egiten da.
     ```bash
       java -jar parametroEkorketaSVM.jar train_split_BOW_FSS.arff dev_split_BOW_FSS.arff SVM.model paramOptimo.txt
     ```
   
  3. kalitatearenEstimazioa.jar: Support Vector Machine teknika erabiltzen da programa honetan. Parametro optimoekin modelo optimoa lortuko da.
    ```bash
      java -jar kalitatearenEstimazioa.jar train_split_BOW_FSS.arff dev_split_BOW_FSS.arff SVMopt.model paramOptimo.txt ebaluazioSVM.txt
    ```

3. Iragarpenak:
   
  1. sailkapena.jar: Test multzoko iragarpenak egingo dituen programa da.
     ```bash
       java -jar sailkapena.jar SVMopt.model test_BOW_FSS.arff  predictions.txt
     ```
     
Exekutagarri guztietan, argumenturik gabe exekutatuz gero edo beharrezko argumentu kopuruarekin bat egiten ez badu, errore mezua aterako da, eta bertan argumentuen zehaztapenak azaltzen dira.

