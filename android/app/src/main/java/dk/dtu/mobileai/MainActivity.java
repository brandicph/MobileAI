package dk.dtu.mobileai;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.telephony.CellInfo;
import android.telephony.CellInfoLte;
import android.telephony.CellSignalStrength;
import android.telephony.CellSignalStrengthLte;
import android.telephony.PhoneStateListener;
import android.telephony.SignalStrength;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.NameValuePair;
import org.apache.http.StatusLine;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.net.HttpURLConnection;
import java.net.URL;
import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;

public class MainActivity extends AppCompatActivity {

    private final static String LTE_TAG = "LTE_Tag";
    private final static String LTE_SIGNAL_STRENGTH = "getLteSignalStrength";
    private final static int MY_PERMISSIONS_REQUEST = 0;
    private static final String[] SIGNAL_STRENGTH_NAMES = {
            "none", "poor", "moderate", "good", "great"
    };

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    private ScrollView scroller;
    private TextView console;
    private Button downloadButton;
    private ProgressBar downloadProgress;
    private SignalStrength mSignalStrength;
    private CellSignalStrengthLte mCellSignalStrength;
    private TelephonyManager telephonyManager;
    private PhoneStateListener mListener;

    public double ping(String url) {
        double result = -1.0;
        try {
            java.lang.Process process = Runtime.getRuntime().exec(
                    "ping -c 1 " + url);
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    process.getInputStream()));
            int i;
            char[] buffer = new char[4096];
            StringBuffer output = new StringBuffer();
            String op[] = new String[64];
            String delay[] = new String[8];
            while ((i = reader.read(buffer)) > 0)
                output.append(buffer, 0, i);
            reader.close();
            op = output.toString().split("\n");
            delay = op[1].split("time=");

            // body.append(output.toString()+"\n");
            Log.d("Pinger", "Ping: " + delay[0] + ", " + delay[1]);
            result = Double.parseDouble(delay[1].replace(" ms", ""));

        } catch (IOException e) {
            // body.append("Error\n");
            return result;
        }
        return result;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        telephonyManager = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);
        scroller = (ScrollView) findViewById(R.id.scroller);
        console = (TextView) findViewById(R.id.console);
        downloadButton = (Button) findViewById(R.id.downloadBtn);
        downloadProgress = (ProgressBar) findViewById(R.id.downloadProgress);

        downloadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final DownloadTask downloadTask = new DownloadTask(MainActivity.this);
                downloadTask.execute("http://ipv6.download.thinkbroadband.com/10MB.zip");
            }
        });

        // Listener for the signal strength.
        // https://www.programcreek.com/java-api-examples/index.php?api=android.telephony.CellInfo
        mListener = new PhoneStateListener()
        {
            @Override
            public void onSignalStrengthsChanged(SignalStrength signalStrength)
            {
                mSignalStrength = signalStrength;

                DateFormat date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
                String dateString = date.format(new Date().getTime());

                MainActivity.this.print("--- " + dateString);

                int lteSignalStrength = invokeSignalStrength("getLteSignalStrength");
                int lteRssi = invokeSignalStrength("getLteDbm");
                int lteRsrp = invokeSignalStrength("getLteRsrp");
                int lteRsrpBoost = invokeSignalStrength("getLteRsrpBoost");
                int lteRsrq = invokeSignalStrength("getLteRsrq");
                int lteRssnr = invokeSignalStrength("getLteRssnr");
                int lteCqi = invokeSignalStrength("getLteCqi");
                int lteLevel = invokeSignalStrength("getLteLevel");
                //String lteLevelStatus = SIGNAL_STRENGTH_NAMES[lteLevel];
                //MainActivity.this.print(lteLevelStatus);
                //getLTECqi();
                //getLTEsignalStrength();

                double pingResult = MainActivity.this.ping("www.google.com");
                MainActivity.this.print("phone : ping = " + pingResult + " ms");

                Log.d(LTE_TAG, mSignalStrength.toString());
                try {
                    List<CellInfo> cellInfoList = telephonyManager.getAllCellInfo();
                    for (CellInfo cellInfo : cellInfoList)
                    {
                        if (cellInfo instanceof CellInfoLte)
                        {
                            // cast to CellInfoLte and call all the CellInfoLte methods you need
                            mCellSignalStrength = ((CellInfoLte)cellInfo).getCellSignalStrength();
                            //Log.d(LTE_TAG, mCellSignalStrength.toString());
                            break;
                        }
                    }
                } catch (SecurityException ex) {
                    Log.e(LTE_TAG, "Security Permission", ex);
                }


            }

        };


        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WAKE_LOCK) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{ Manifest.permission.ACCESS_COARSE_LOCATION, Manifest.permission.WAKE_LOCK, Manifest.permission.WRITE_EXTERNAL_STORAGE }, MY_PERMISSIONS_REQUEST);
        } else {
            Log.i(LTE_TAG, "Permission granted");
            // Register the listener for the telephony manager
            telephonyManager.listen(mListener, PhoneStateListener.LISTEN_SIGNAL_STRENGTHS | PhoneStateListener.LISTEN_CELL_INFO | PhoneStateListener.LISTEN_DATA_CONNECTION_STATE);

        }
        // Example of a call to a native method
        // TextView tv = (TextView) findViewById(R.id.sample_text);
        // tv.setText(stringFromJNI());
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    Log.i(LTE_TAG, "Permission granted");
                    // Register the listener for the telephony manager
                    telephonyManager.listen(mListener, PhoneStateListener.LISTEN_SIGNAL_STRENGTHS | PhoneStateListener.LISTEN_CELL_INFO | PhoneStateListener.LISTEN_DATA_CONNECTION_STATE);

                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    private int invokeSignalStrength(String method) {
        try
        {
            Method[] methods = android.telephony.SignalStrength.class.getMethods();

            for (Method mthd : methods)
            {
                if (mthd.getName().equals(method))
                {
                    int LTEsignalStrength = (Integer) mthd.invoke(mSignalStrength, new Object[] {});
                    this.print("phone : " + method + " = " + LTEsignalStrength);
                    return LTEsignalStrength;
                }
            }
        }
        catch (Exception e)
        {
            Log.e(LTE_TAG, "Exception: " + e.toString());
        }
        return Integer.MAX_VALUE;
    }

    private int invokeCellSignalStrength(String method){
        try
        {
            Method[] methods = android.telephony.SignalStrength.class.getMethods();

            for (Method mthd : methods)
            {
                Log.d(LTE_TAG, "method: " + mthd);
                if (mthd.getName().equals(method))
                {
                    int LTEsignalStrength = (Integer) mthd.invoke(mCellSignalStrength, new Object[] {});
                    this.print("cell : " + method + " = " + LTEsignalStrength);
                    return LTEsignalStrength;
                }
            }
        }
        catch (Exception e)
        {
            Log.e(LTE_TAG, "Exception: " + e.toString());
        }
        return Integer.MAX_VALUE;
    }

    private void getLTECqi() {
        try
        {
            /*
            part[0] = "Signalstrength:"  _ignore this, it's just the title_

            parts[1] = GsmSignalStrength

            parts[2] = GsmBitErrorRate

            parts[3] = CdmaDbm

            parts[4] = CdmaEcio

            parts[5] = EvdoDbm

            parts[6] = EvdoEcio

            parts[7] = EvdoSnr

            parts[8] = LteSignalStrength

            parts[9] = LteRsrp

            parts[10] = LteRsrq

            parts[11] = LteRssnr

            parts[12] = LteCqi

            parts[13] = gsm|lte|cdma

            parts[14] = _not really sure what this number is_
             */
            String ssignal = mSignalStrength.toString();
            String[] parts = ssignal.split(" ");
            String lteCqi = parts[12];

            Log.i(LTE_TAG, "LteCqi = " + lteCqi);
            Log.i(LTE_TAG, "ALL == " + ssignal);
            print("LteCqi = " + lteCqi);
            //print("ALL == " + ssignal);

            //new RetrieveFeedTask().execute();
            //new PostTask().execute("value");
            /*
            MobileAiApi mobileAiApi = MobileAiApi.retrofit.create(MobileAiApi.class);
            Call<List<CellInfoLte>> call = mobileAiApi.getAllCellInfoLte();

            call.enqueue(new Callback<List<CellInfoLte>>() {
                @Override
                public void onResponse(Call<List<CellInfoLte>> call, Response<List<CellInfoLte>> response) {
                    List<CellInfoLte> result = response.body();
                    Log.i(LTE_TAG, result.toString());
                }

                @Override
                public void onFailure(Call<List<CellInfoLte>> call, Throwable t) {

                }

            });
            */

            final MobileAiApi mobileAiApi = MobileAiApi.retrofit.create(MobileAiApi.class);

            try {
                List<CellInfo> cellInfo = telephonyManager.getAllCellInfo();
                for (final CellInfo ci : cellInfo){
                    if (ci instanceof android.telephony.CellInfoLte){
                        /*
                        DateFormat date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
                        Call<CellInfoLte> call = mobileAiApi.postCellInfoLte(ci.isRegistered(), date.format(ci.getTimeStamp() / 1000));

                        call.enqueue(new Callback<CellInfoLte>() {
                            @Override
                            public void onResponse(Call<CellInfoLte> call, Response<CellInfoLte> response) {
                                final CellInfoLte result = response.body();
                                final String url = result.getUrl();

                                android.telephony.CellIdentityLte cellIdentityLte = ((android.telephony.CellInfoLte) ci).getCellIdentity();
                                Call<CellIdentityLte> callCellIdentityLte = mobileAiApi.postCellIdentityLte(
                                        cellIdentityLte.getMcc(),
                                        cellIdentityLte.getTac(),
                                        cellIdentityLte.getEarfcn(),
                                        cellIdentityLte.getCi(),
                                        cellIdentityLte.getMnc(),
                                        cellIdentityLte.getPci(),
                                        url);

                                android.telephony.CellSignalStrengthLte cellSignalStrengthLte = ((android.telephony.CellInfoLte) ci).getCellSignalStrength();

                                Call<CellSignalStrengthLte> callCellSignalStrengthLte = mobileAiApi.postCellSignalStrengthLte(
                                        //cellSignalStrengthLte.getCqi(),
                                        2147483647,
                                        //cellSignalStrengthLte.getRssnr(),
                                        2147483647,
                                        cellSignalStrengthLte.getTimingAdvance(),
                                        cellSignalStrengthLte.getDbm(),
                                        //cellSignalStrengthLte.getRsrp(),
                                        2147483647,
                                        //cellSignalStrengthLte.getRsrq(),
                                        2147483647,
                                        url);

                                callCellIdentityLte.enqueue(new Callback<CellIdentityLte>() {

                                    @Override
                                    public void onResponse(Call<CellIdentityLte> call, Response<CellIdentityLte> response) {

                                    }

                                    @Override
                                    public void onFailure(Call<CellIdentityLte> call, Throwable t) {

                                    }
                                });

                                callCellSignalStrengthLte.enqueue(new Callback<CellSignalStrengthLte>() {
                                    @Override
                                    public void onResponse(Call<CellSignalStrengthLte> call, Response<CellSignalStrengthLte> response) {

                                    }

                                    @Override
                                    public void onFailure(Call<CellSignalStrengthLte> call, Throwable t) {

                                    }
                                });


                            }

                            @Override
                            public void onFailure(Call<CellInfoLte> call, Throwable t) {
                                Log.i(LTE_TAG, t.toString());
                            }

                        });
                        */
                    }
                    Log.i(LTE_TAG, "=============> " + ci.toString());
                }
            } catch (SecurityException ex){
                Log.e(LTE_TAG, "Security Permission", ex);
            }

        }
        catch (Exception e)
        {
            Log.e(LTE_TAG, "Exception: " + e.toString());
        }
    }

    private void print(String text){
        console.append(text);
        console.append("\n");
        scroller.fullScroll(scroller.FOCUS_DOWN);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();


    // usually, subclasses of AsyncTask are declared inside the activity class.
// that way, you can easily modify the UI thread from here
    class DownloadTask extends AsyncTask<String, Integer, String> {

        private Context context;
        private PowerManager.WakeLock mWakeLock;

        public DownloadTask(Context context) {
            this.context = context;
        }

        @Override
        protected String doInBackground(String... sUrl) {
            InputStream input = null;
            OutputStream output = null;
            HttpURLConnection connection = null;
            try {
                Log.d("DOWNLOAD", "Starting");
                URL url = new URL(sUrl[0]);
                connection = (HttpURLConnection) url.openConnection();
                connection.connect();

                // expect HTTP 200 OK, so we don't mistakenly save error report
                // instead of the file
                if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
                    String msg = "Server returned HTTP " + connection.getResponseCode()
                            + " " + connection.getResponseMessage();
                    Log.d("DOWNLOAD", msg);
                    return msg;
                }

                // this will be useful to display download percentage
                // might be -1: server did not report the length
                int fileLength = connection.getContentLength();

                // download the file
                input = connection.getInputStream();

                //String path = Environment.DIRECTORY_DOWNLOADS + "file.zip";
                //Log.d("DOWNLOAD", path);

                //output = new FileOutputStream(path);

                byte data[] = new byte[4096];
                long total = 0;
                int count;
                while ((count = input.read(data)) != -1) {
                    // allow canceling with back button
                    if (isCancelled()) {
                        input.close();
                        return null;
                    }
                    total += count;
                    // publishing the progress....
                    if (fileLength > 0) {// only if total length is known
                        int status = (int) (total * 100 / fileLength);
                        publishProgress(status);
                        Log.d("DOWNLOAD", "Progress: " + status);
                    }
                    //output.write(data, 0, count);
                }
            } catch (Exception e) {
                Log.e("DOWNLOAD", e.toString());
                return e.toString();
            } finally {

                try {
                /*if (output != null)
                    output.close();*/
                    if (input != null)
                        input.close();
                } catch (IOException ignored) {
                }

                if (connection != null)
                    connection.disconnect();
            }
            return null;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            // take CPU lock to prevent CPU from going off if the user
            // presses the power button during download
            PowerManager pm = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
            mWakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,
                    getClass().getName());
            mWakeLock.acquire();
        }

        @Override
        protected void onProgressUpdate(Integer... progress) {
            super.onProgressUpdate(progress);
            // if we get here, length is known, now set indeterminate to false
            downloadProgress.setIndeterminate(false);
            downloadProgress.setMax(100);
            downloadProgress.setProgress(progress[0]);
        }

        @Override
        protected void onPostExecute(String result) {
            mWakeLock.release();
            if (result != null)
                Toast.makeText(context,"Download error: "+result, Toast.LENGTH_LONG).show();
            else
                Toast.makeText(context,"File downloaded", Toast.LENGTH_SHORT).show();
        }
    }
}



class RetrieveFeedTask extends AsyncTask<Void, Void, String> {

    private Exception exception;

    protected void onPreExecute() {

    }

    protected String doInBackground(Void... urls) {

        try {
            URL url = new URL("http://192.168.1.4:8000");
            HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
            try {
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
                StringBuilder stringBuilder = new StringBuilder();
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    stringBuilder.append(line).append("\n");
                }
                bufferedReader.close();
                return stringBuilder.toString();
            }
            finally{
                urlConnection.disconnect();
            }
        }
        catch(Exception e) {
            Log.e("ERROR", e.getMessage(), e);
            return null;
        }
    }

    protected void onPostExecute(String response) {
        if(response == null) {
            response = "THERE WAS AN ERROR";
        }
        Log.i("INFO", response);
    }
}


class PostTask extends AsyncTask<String, String, String> {
    @Override
    protected String doInBackground(String... data) {
        // Create a new HttpClient and Post Header
        HttpClient httpclient = new DefaultHttpClient();
        HttpPost httppost = new HttpPost("http://192.168.1.4:8000/cellinfolte/");

        try {
            //add data
            List<NameValuePair> nameValuePairs = new ArrayList<NameValuePair>(1);
            nameValuePairs.add(new BasicNameValuePair("registered", data[0]));
            nameValuePairs.add(new BasicNameValuePair("timestamp", data[0]));
            httppost.setEntity(new UrlEncodedFormEntity(nameValuePairs));
            //execute http post
            HttpResponse response = httpclient.execute(httppost);

        } catch (ClientProtocolException e) {

        } catch (IOException e) {

        }
        return null;
    }
}