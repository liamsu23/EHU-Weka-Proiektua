import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class ApplyFssToDev {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Mesedez, atributua ondo sartu.");
            return;
        }

        String inputDevBowPath = args[0];
        String outputDevBowFssPath = args[1];

        try {
            // 1️⃣ Cargar conjunto de desarrollo
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(inputDevBowPath);
            Instances devData = source.getDataSet();

            // Verificar si la clase está definida
            if (devData.classIndex() == -1) {
                devData.setClassIndex(devData.attribute("Cause_of_Death").index());
            }

            System.out.println("📊 Número de atributos antes de la selección: " + devData.numAttributes());

            // 2️⃣ Cargar el filtro de selección de atributos guardado
            FileInputStream fis = new FileInputStream("fss_filter.model");
            ObjectInputStream ois = new ObjectInputStream(fis);
            Filter filter = (Filter) ois.readObject();
            ois.close();
            fis.close();

            // 3️⃣ Aplicar el filtro a dev
            Instances selectedDevData = Filter.useFilter(devData, filter);

            System.out.println("✅ Número de atributos después de la selección: " + selectedDevData.numAttributes());

            // 4️⃣ Guardar el nuevo conjunto de datos en dev_fss.arff
            ArffSaver saver = new ArffSaver();
            saver.setInstances(selectedDevData);
            saver.setFile(new File(outputDevBowFssPath));
            saver.writeBatch();

            System.out.println("🚀 Conjunto de desarrollo con FSS aplicado guardado en " + outputDevBowFssPath);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }
}