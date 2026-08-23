import java.util.Random;

/**
 * 实时预测冒烟测试：加载 ONNX 模型跑单次推理，验证原生库与耗时。
 * Usage: java -cp ".:lib/*" PredSmoke [model.onnx]
 */
public class PredSmoke {
    public static void main(String[] args) {
        String path = args.length > 0 ? args[0] : "train/last.onnx";
        WatchHandServer.loadPredictor(path);
        float[][] orig = new float[60][96];
        Random r = new Random(0);
        for (int d = 0; d < 60; d++) for (int t = 0; t < 96; t++) orig[d][t] = r.nextFloat() * 100;
        WatchHandServer.predictClass(orig);  // 预热
        long t0 = System.nanoTime();
        int p = WatchHandServer.predictClass(orig);
        System.out.println("pred=" + p + " latency=" + ((System.nanoTime() - t0) / 1000000) + "ms");
    }
}
