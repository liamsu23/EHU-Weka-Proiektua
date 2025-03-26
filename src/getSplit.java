import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;

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

        Randomize randomFilter = new Randomize();
        randomFilter.setRandomSeed(1);
        randomFilter.setInputFormat(data);
        Instances randomData = Filter.useFilter(data, randomFilter);

        // 2. Primero obtener TRAIN (80%)
        RemovePercentage removeFilter = new RemovePercentage();
        removeFilter.setInputFormat(data);
        removeFilter.setPercentage(80); // Eliminar 20%
        removeFilter.setInvertSelection(true); // Conservar 80%
        Instances trainData = Filter.useFilter(randomData, removeFilter);

        // 3. Luego obtener DEV (20%) del original
        removeFilter = new RemovePercentage(); // Reiniciar filtro
        removeFilter.setInputFormat(data);
        removeFilter.setPercentage(80); // Eliminar 80%
        removeFilter.setInvertSelection(false); // Conservar 20%
        Instances devData = Filter.useFilter(randomData, removeFilter);

        // 4. Verificación
        System.out.println("Total instancias: " + data.numInstances());
        System.out.println("Train (80%): " + trainData.numInstances());
        System.out.println("Dev (20%): " + devData.numInstances());

        // 5. Guardar
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainData);
        saver.setFile(new File(outTrainSplitPath));
        saver.writeBatch();

        saver.setInstances(devData);
        saver.setFile(new File(outDevSplitPath));
        saver.writeBatch();
    }
}