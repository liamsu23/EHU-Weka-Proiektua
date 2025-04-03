import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.supervised.instance.Resample;

import java.io.File;


/**
 * Klase honek ARFF formatuko datu-multzoak banaketa egoki batean partizionatzeko
 * funtzionalitateak ematen ditu, machine learning ereduen entrenamendurako eta garapenerako.
 *
 * <p>Prozesu honek hurrengo urratsak egiten ditu:</p>
 * <ol>
 *   <li>Datu-multzo originala kargatzen du</li>
 *   <li>Klase-desoreka konpontzen du Resample filtroa aplikatuz (biasToUniformClass)</li>
 *   <li>Datuak ausaz permutatzen ditu (Randomize)</li>
 *   <li>Datuak bi multzotan banatzen ditu:
 *     <ul>
 *       <li>Entrenamendurako multzoa (80%)</li>
 *       <li>Garapenerako multzoa (20%)</li>
 *     </ul>
 *   </li>
 * </ol>
 *
 * <p>Programak hiru argumentu hauek behar ditu:</p>
 * <ol>
 *   <li>train_RAW.arff: Sarrerako ARFF fitxategiaren path-a (datu-multzo osoa)</li>
 *   <li>train_split_RAW.arff: Irteerako entrenamendu ARFF fitxategiaren path-a (80%)</li>
 *   <li>dev_split_RAW.arff: Irteerako garapen ARFF fitxategiaren path-a (20%)</li>
 * </ol>
 *
 * <p>Klasearen ezaugarri nagusiak:</p>
 * <ul>
 *   <li>Klase-balantzea: biasToUniformClass parametroa erabiliz klase-minoritarioen pisua handitzen du</li>
 *   <li>Erreproduzigarritasuna: RandomSeed finkoak erabiliz emaitza berberak lortzeko</li>
 *   <li>Banaketa proportzionala: 80/20 erlazioa mantentzen du banaketan</li>
 * </ul>
 *
 * <p>Oharra: Klaseak Weka liburutegiaren Instantziak eta Filter klaseak erabiltzen ditu
 * datuak prozesatzeko eta manipulatzeko.</p>
 */

public class getSplit {
    public static void main(String[] args) throws Exception {
        // Validación de argumentos
        if (args.length != 3) {
            System.out.println("Uso: java getSplit <train_RAW.arff> <train_split_RAW.arff> <dev_split_RAW.arff>");
            return;
        }

        String inTrainPath = args[0];
        String outTrainSplitPath = args[1];
        String outDevSplitPath = args[2];

        // 1. Cargar datos
        DataSource source = new DataSource(inTrainPath);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // 2. Aplicar Resample con biasToUniformClass
        Resample resampleFilter = new Resample();
        resampleFilter.setBiasToUniformClass(1.0); // Balancear clases
        resampleFilter.setNoReplacement(false); // Permitir reemplazo
        resampleFilter.setSampleSizePercent(100); // Mantener mismo tamaño de dataset
        resampleFilter.setRandomSeed(1);
        resampleFilter.setInputFormat(data);
        Instances balancedData = Filter.useFilter(data, resampleFilter);

        // 3. Aleatorizar los datos balanceados
        Randomize randomFilter = new Randomize();
        randomFilter.setRandomSeed(1);
        randomFilter.setInputFormat(balancedData);
        Instances randomData = Filter.useFilter(balancedData, randomFilter);

        // 4. Obtener TRAIN (80%)
        RemovePercentage removeFilter = new RemovePercentage();
        removeFilter.setInputFormat(randomData);
        removeFilter.setPercentage(80);
        removeFilter.setInvertSelection(true);
        Instances trainData = Filter.useFilter(randomData, removeFilter);

        // 5. Obtener DEV (20%)
        removeFilter = new RemovePercentage();
        removeFilter.setInputFormat(randomData);
        removeFilter.setPercentage(80);
        removeFilter.setInvertSelection(false);
        Instances devData = Filter.useFilter(randomData, removeFilter);

        // 6. Verificación
        System.out.println("Total instancias (balanceadas): " + balancedData.numInstances());
        System.out.println("Train (80%): " + trainData.numInstances());
        System.out.println("Dev (20%): " + devData.numInstances());

        // 7. Guardar
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainData);
        saver.setFile(new File(outTrainSplitPath));
        saver.writeBatch();

        saver.setInstances(devData);
        saver.setFile(new File(outDevSplitPath));
        saver.writeBatch();
    }
}
