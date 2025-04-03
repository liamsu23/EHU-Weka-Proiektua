import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;


/**
 * Klase honek SVM (Support Vector Machine) eredu baten hiperparametroen optimizazio aurreratua
 * egiten du, kernel mota desberdinak alderatuz (RBF, Polinomial eta PUK) eta haien parametroak
 * optimizatuz F1-macro metrikaren arabera.
 *
 * <p>Funtzionamendu nagusia:</p>
 * <ol>
 *   <li>Datu-multzoak kargatu (train eta dev FSS formatuan)</li>
 *   <li>Kernel mota desberdinen konbinazioak probatu:
 *     <ul>
 *       <li>RBF Kernel: gamma parametroarekin</li>
 *       <li>Polinomial Kernel: degree parametroarekin</li>
 *       <li>PUK Kernel: omega eta sigma parametroekin</li>
 *     </ul>
 *   </li>
 *   <li>Eredu onena gorde eta bere parametroak fitxategian bildu</li>
 * </ol>
 *
 * <p>Erabilera:</p>
 * <pre>java parametroEkorketaSVM_3 train_split_BOW_FSS.arff dev_split_BOW_FSS.arff SVM_3.model parameters_3.txt</pre>
 *
 * <p>Bilatzen diren parametroak:</p>
 * <ul>
 *   <li>C: Penalizazio-faktorea (1, 10, 100, 1000)</li>
 *   <li>RBF: Gamma (0.01, 0.1, 1)</li>
 *   <li>Polinomial: Degree (2, 3, 4)</li>
 *   <li>PUK: Omega (0.01, 0.1, 1) eta Sigma (0.01, 0.1, 1)</li>
 * </ul>
 *
 * <p>Outputak:</p>
 * <ul>
 *   <li>.model fitxategia: Entrenatutako SVM eredu hoberena</li>
 *   <li>.txt fitxategia: Aurkikitako parametro optimoak (kernel mota barne)</li>
 * </ul>
 */

public class parametroEkorketaSVM_3 {
    static double bestFMeasure = 0;
    static double bestC = 0, bestGamma = 0, bestDegree = 0, bestOmega = 0, bestSigma = 0;
    static String bestKernel = "";
    static SMO bestModel = null;

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println("Uso: java SVM <train.arff> <dev.arff> <model.model> <output.txt>");
            return;
        }
        try {
            String inTrainFSSPath = args[0];
            String inDevFSSPath = args[1];
            String outModelPath = args[2];
            String outputFilePath = args[3];

            // Cargar datasets
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(inTrainFSSPath);
            Instances dataTrain = source.getDataSet();

            ConverterUtils.DataSource sourceDev = new ConverterUtils.DataSource(inDevFSSPath);
            Instances dataDev = sourceDev.getDataSet();

            if (dataTrain.classIndex() == -1) {
                dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
            }
            if (dataDev.classIndex() == -1) {
                dataDev.setClassIndex(dataDev.numAttributes() - 1);
            }

            // Parámetros a explorar
            double[] cValues = {1, 10, 100, 1000};
            double[] gammaValues = {0.01, 0.1, 1};
            int[] polyDegrees = {2, 3, 4};
            double[] omegaValues = {0.01, 0.1, 1};
            double[] sigmaValues = {0.01, 0.1, 1};
            Kernel[] kernels = {new RBFKernel(), new PolyKernel(), new Puk()};

            System.out.println("🔍 Búsqueda de hiperparámetros");
            for (Kernel kernel : kernels) {
                for (double c : cValues) {
                    if (kernel instanceof PolyKernel) {
                        for (int degree : polyDegrees) {
                            ((PolyKernel) kernel).setExponent(degree);
                            testKernel(dataTrain, dataDev, kernel, c, 0, degree, 0, 0);
                        }
                    } else if (kernel instanceof RBFKernel) {
                        for (double gamma : gammaValues) {
                            ((RBFKernel) kernel).setGamma(gamma);
                            testKernel(dataTrain, dataDev, kernel, c, gamma, 0, 0, 0);
                        }
                    } else if (kernel instanceof Puk) {
                        for (double omega : omegaValues) {
                            for (double sigma : sigmaValues) {
                                ((Puk) kernel).setOmega(omega);
                                ((Puk) kernel).setSigma(sigma);
                                testKernel(dataTrain, dataDev, kernel, c, 0, 0, omega, sigma);
                            }
                        }
                    } else {
                        testKernel(dataTrain, dataDev, kernel, c, 0, 0, 0, 0);
                    }
                }
            }

            // Guardar el mejor modelo encontrado
            if (bestModel != null) {
                SerializationHelper.write(outModelPath, bestModel);
                System.out.println("✅ Modelo guardado en: " + outModelPath);
            }

            // Guardar los parámetros en un archivo
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath));
            writer.write("Train=" + inTrainFSSPath + "\n");
            writer.write("Dev=" + inDevFSSPath + "\n");
            writer.write("Eredua=" + outModelPath + "\n");
            writer.write("\n");
            writer.write("Kernel=" + bestKernel + "\n");
            writer.write("C=" + bestC + "\n");
            writer.write("Gamma=" + bestGamma + "\n");
            writer.write("Degree=" + bestDegree + "\n");
            writer.write("Omega=" + bestOmega + "\n");
            writer.write("Sigma=" + bestSigma + "\n");
            writer.close();

            System.out.println("Parámetros guardados en: " + outputFilePath);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    private static void testKernel(Instances dataTrain, Instances dataDev, Kernel kernel, double c, double gamma, int degree, double omega, double sigma) throws Exception {
        SMO svm = new SMO();
        svm.setKernel(kernel);
        svm.setC(c);

        svm.buildClassifier(dataTrain);
        Evaluation eval = new Evaluation(dataTrain);
        eval.evaluateModel(svm, dataDev);

        double f1Macro = 0.0;
        for (int i = 0; i < dataTrain.numClasses(); i++) {
            f1Macro += eval.fMeasure(i);
        }
        f1Macro /= dataTrain.numClasses();

        System.out.println("Kernel=" + kernel.getClass().getSimpleName() + ", C=" + c +
                (kernel instanceof PolyKernel ? ", Degree=" + degree : "") +
                (kernel instanceof RBFKernel ? ", Gamma=" + gamma : "") +
                (kernel instanceof Puk ? ", Omega=" + omega + ", Sigma=" + sigma : "") +
                ", F1-macro=" + f1Macro);

        if (!Double.isNaN(f1Macro) && f1Macro > bestFMeasure) {
            bestFMeasure = f1Macro;
            bestKernel = kernel.getClass().getSimpleName();
            bestC = c;
            if (kernel instanceof RBFKernel) bestGamma = gamma;
            if (kernel instanceof PolyKernel) bestDegree = degree;
            if (kernel instanceof Puk) {
                bestOmega = omega;
                bestSigma = sigma;
            }
            bestModel = svm;
        }
    }
}
