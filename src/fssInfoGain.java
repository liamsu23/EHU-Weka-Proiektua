import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

public class fssInfoGain {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Mesedez, atributua ondo sartu.");
            return;
        }

        String inputTrainBowPath = args[0];
        String outputTrainBowFssPath = args[1];

        try {
            // 1️⃣ Cargar conjunto de entrenamiento
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(inputTrainBowPath);
            Instances data = source.getDataSet();

            // Verificar si la clase está definida
            if (data.classIndex() == -1) {
                data.setClassIndex(data.attribute("Cause_of_Death").index());
            }

            System.out.println("📊 Número de atributos antes de la selección: " + data.numAttributes());

            // 2️⃣ Aplicar selección de atributos usando InfoGainAttributeEval + Ranker
            AttributeSelection filter = new AttributeSelection();
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval(); // Evalúa ganancia de información
            Ranker search = new Ranker(); // Ordena los atributos de mayor a menor importancia
            search.setNumToSelect(1000); // Mantiene todos los atributos relevantes (puedes cambiarlo)

            filter.setEvaluator(evaluator);
            filter.setSearch(search);
            filter.setInputFormat(data);
            Instances selectedData = Filter.useFilter(data, filter);

            System.out.println("✅ Número de atributos después de la selección: " + selectedData.numAttributes());

            // 3️⃣ Guardar el nuevo conjunto de datos en train_fss.arff
            ArffSaver saver = new ArffSaver();
            saver.setInstances(selectedData);
            saver.setFile(new File(outputTrainBowFssPath));
            saver.writeBatch();

            // 4️⃣ Guardar el filtro de selección de atributos para reutilizarlo en dev
            FileOutputStream fos = new FileOutputStream("fss_filter.model");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(filter);
            oos.close();
            fos.close();

            System.out.println("🚀 Filtro de selección de atributos guardado en fss_filter.model");

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }
}