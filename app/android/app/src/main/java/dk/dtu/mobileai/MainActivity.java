package dk.dtu.mobileai;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.telephony.CellInfo;
import android.telephony.CellInfoLte;
import android.telephony.PhoneStateListener;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;
import static android.Manifest.permission.READ_PHONE_STATE;
/*
TS 27.007 8.5
Defined values
<rsrp>:
0 -113 dBm or less
1 -111 dBm
2...30 -109... -53 dBm
31 -51 dBm or greater
99 not known or not detectable
 */

/*
The parts[] array will then contain these elements:

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
parts[13] = gsm|lte
parts[14] = _not reall sure what this number is_
 */

public class MainActivity extends AppCompatActivity {

    public static final int REQUEST_ID_MULTIPLE_PERMISSIONS = 124;

    SignalStrengthListener signalStrengthListener;
    LocationService locationService;

    TextView signalStrengthTextView, signalStrengthTextView2;
    TextView cellIDTextView;
    TextView cellMccTextView;
    TextView cellMncTextView;
    TextView cellPciTextView;
    TextView cellTacTextView;
    TextView ueDeviceNameTextView;
    TextView ueImeiTextView;
    TextView ueImsiTextView;
    TextView locLatitudeTextView;
    TextView locLongitudeTextView;

    List<CellInfo> cellInfoList;
    int cellSig, cellID, cellMcc, cellMnc, cellPci, cellTac = 0;
    String ueIMSI, ueIMEI, ueDeviceName = null;

    public double longitude;
    public double latitude;

    TelephonyManager tm;

    LocationManager lm;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //setup content stuff
        this.setContentView(R.layout.activity_main);

        signalStrengthTextView = (TextView) findViewById(R.id.signalStrengthTextView);
        signalStrengthTextView2 = (TextView) findViewById(R.id.signalStrengthTextView2);
        cellIDTextView = (TextView) findViewById(R.id.cellIDTextView);
        cellMccTextView = (TextView) findViewById(R.id.cellMccTextView);
        cellMncTextView = (TextView) findViewById(R.id.cellMncTextView);
        cellPciTextView = (TextView) findViewById(R.id.cellPciTextView);
        cellTacTextView = (TextView) findViewById(R.id.cellTacTextView);
        ueDeviceNameTextView = (TextView) findViewById(R.id.ueDeviceNameTextView);
        ueImeiTextView = (TextView) findViewById(R.id.ueImeiTextView);
        ueImsiTextView = (TextView) findViewById(R.id.ueImsiTextView);
        locLatitudeTextView = (TextView) findViewById(R.id.locLatitudeTextView);
        locLongitudeTextView = (TextView) findViewById(R.id.locLongitudeTextView);


        //start the signal strength listener
        signalStrengthListener = new SignalStrengthListener();

        ((TelephonyManager) getSystemService(TELEPHONY_SERVICE)).listen(signalStrengthListener, SignalStrengthListener.LISTEN_SIGNAL_STRENGTHS);
        tm = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);

        //start the location listener
        locationService = new LocationService(this);

        lm = (LocationManager) this.getSystemService(Context.LOCATION_SERVICE);


        // Adding if condition inside button.

        // If All permission is enabled successfully then this block will execute.
        if(CheckingPermissionIsEnabledOrNot())
        {
            Toast.makeText(MainActivity.this, "All Permissions Granted Successfully", Toast.LENGTH_LONG).show();
            try {
                cellInfoList = tm.getAllCellInfo();
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 2000,10, locationService);
            } catch (SecurityException e){
                RequestMultiplePermission();
            } catch (Exception e) {
                Log.d("SignalStrength", "+++++++++++++++++++++++++++++++++++++++++ null array spot 1: " + e);

            }
        }

        // If, If permission is not enabled then else condition will execute.
        else {

            //Calling method to enable permission.
            RequestMultiplePermission();

        }




        try {
            for (CellInfo cellInfo : cellInfoList) {
                if (cellInfo instanceof CellInfoLte) {
                    // cast to CellInfoLte and call all the CellInfoLte methods you need
                    // gets RSRP cell signal strength:
                    cellSig = ((CellInfoLte) cellInfo).getCellSignalStrength().getDbm();

                    // Gets the LTE cell indentity: (returns 28-bit Cell Identity, Integer.MAX_VALUE if unknown)
                    cellID = ((CellInfoLte) cellInfo).getCellIdentity().getCi();

                    // Gets the LTE MCC: (returns 3-digit Mobile Country Code, 0..999, Integer.MAX_VALUE if unknown)
                    cellMcc = ((CellInfoLte) cellInfo).getCellIdentity().getMcc();

                    // Gets theLTE MNC: (returns 2 or 3-digit Mobile Network Code, 0..999, Integer.MAX_VALUE if unknown)
                    cellMnc = ((CellInfoLte) cellInfo).getCellIdentity().getMnc();

                    // Gets the LTE PCI: (returns Physical Cell Id 0..503, Integer.MAX_VALUE if unknown)
                    cellPci = ((CellInfoLte) cellInfo).getCellIdentity().getPci();

                    // Gets the LTE TAC: (returns 16-bit Tracking Area Code, Integer.MAX_VALUE if unknown)
                    cellTac = ((CellInfoLte) cellInfo).getCellIdentity().getTac();

                    // Gets the IMEI
                    ueIMEI = tm.getImei();

                    // Gets the IMSI
                    ueIMSI = tm.getSubscriberId();

                    // Gets Device Model Name
                    ueDeviceName = getDeviceName();

                }

            }
        } catch (SecurityException e){

        } catch (Exception e) {
            Log.d("SignalStrength", "++++++++++++++++++++++ null array spot 2: " + e);
        }

    }

    public void updateCoordinates(){
        Log.d("LocationUpdate", "latitude: " + latitude + " longitude: " + longitude);
        locLatitudeTextView.setText(String.valueOf(latitude));
        locLongitudeTextView.setText(String.valueOf(longitude));
    }

    public String getDeviceName() {
        String manufacturer = android.os.Build.MANUFACTURER;
        String model = android.os.Build.MODEL;
        if (model.startsWith(manufacturer)) {
            return capitalize(model);
        } else {
            return capitalize(manufacturer) + " " + model;
        }
    }


    private String capitalize(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        char first = s.charAt(0);
        if (Character.isUpperCase(first)) {
            return s;
        } else {
            return Character.toUpperCase(first) + s.substring(1);
        }
    }


    @Override
    public void onPause() {
        super.onPause();

        try{
            if(signalStrengthListener != null){tm.listen(signalStrengthListener, SignalStrengthListener.LISTEN_NONE);}
        }catch(Exception e){
            e.printStackTrace();
        }
    }


    public void onDestroy() {
        super.onDestroy();

        try{
            if(signalStrengthListener != null){tm.listen(signalStrengthListener, SignalStrengthListener.LISTEN_NONE);}
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    //Permission function starts from here
    private void RequestMultiplePermission() {

        // Creating String Array with Permissions.
        ActivityCompat.requestPermissions(MainActivity.this, new String[]
                {
                        ACCESS_FINE_LOCATION,
                        READ_PHONE_STATE
                }, REQUEST_ID_MULTIPLE_PERMISSIONS);

    }

    // Calling override method.
    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {

            case REQUEST_ID_MULTIPLE_PERMISSIONS:

                if (grantResults.length > 0) {

                    boolean AccessFineLocation = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean ReadPhoneState = grantResults[1] == PackageManager.PERMISSION_GRANTED;

                    if (AccessFineLocation && ReadPhoneState) {

                        Toast.makeText(MainActivity.this, "Permission Granted", Toast.LENGTH_LONG).show();
                    }
                    else {
                        Toast.makeText(MainActivity.this,"Permission Denied",Toast.LENGTH_LONG).show();

                    }
                }

                break;
        }
    }

    // Checking permission is enabled or not using function starts from here.
    public boolean CheckingPermissionIsEnabledOrNot() {

        int FirstPermissionResult = ContextCompat.checkSelfPermission(getApplicationContext(), ACCESS_FINE_LOCATION);
        int SecondPermissionResult = ContextCompat.checkSelfPermission(getApplicationContext(), READ_PHONE_STATE);

        return FirstPermissionResult == PackageManager.PERMISSION_GRANTED &&
                SecondPermissionResult == PackageManager.PERMISSION_GRANTED ;
    }


    private class SignalStrengthListener extends PhoneStateListener {
        @Override
        public void onSignalStrengthsChanged(android.telephony.SignalStrength signalStrength) {

            //++++++++++++++++++++++++++++++++++

            ((TelephonyManager) getSystemService(TELEPHONY_SERVICE)).listen(signalStrengthListener, SignalStrengthListener.LISTEN_SIGNAL_STRENGTHS);

            tm = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);

            String ltestr = signalStrength.toString();
            String[] parts = ltestr.split(" ");
            String cellSig2 = parts[9];

            try {
                cellInfoList = tm.getAllCellInfo();
                for (CellInfo cellInfo : cellInfoList) {
                    if (cellInfo instanceof CellInfoLte) {
                        // cast to CellInfoLte and call all the CellInfoLte methods you need
                        // gets RSRP cell signal strength:
                        cellSig = ((CellInfoLte) cellInfo).getCellSignalStrength().getDbm();

                        // Gets the LTE cell identity: (returns 28-bit Cell Identity, Integer.MAX_VALUE if unknown)
                        cellID = ((CellInfoLte) cellInfo).getCellIdentity().getCi();

                        // Gets the LTE MCC: (returns 3-digit Mobile Country Code, 0..999, Integer.MAX_VALUE if unknown)
                        cellMcc = ((CellInfoLte) cellInfo).getCellIdentity().getMcc();

                        // Gets theLTE MNC: (returns 2 or 3-digit Mobile Network Code, 0..999, Integer.MAX_VALUE if unknown)
                        cellMnc = ((CellInfoLte) cellInfo).getCellIdentity().getMnc();

                        // Gets the LTE PCI: (returns Physical Cell Id 0..503, Integer.MAX_VALUE if unknown)
                        cellPci = ((CellInfoLte) cellInfo).getCellIdentity().getPci();

                        // Gets the LTE TAC: (returns 16-bit Tracking Area Code, Integer.MAX_VALUE if unknown)
                        cellTac = ((CellInfoLte) cellInfo).getCellIdentity().getTac();

                        // Gets the IMEI
                        ueIMEI = tm.getImei();

                        // Gets the IMSI
                        ueIMSI = tm.getSubscriberId();

                        // Gets Device Model Name
                        ueDeviceName = getDeviceName();


                    }
                }
            } catch (SecurityException e){
                RequestMultiplePermission();
            } catch (Exception e) {
                Log.d("SignalStrength", "+++++++++++++++++++++++++++++++ null array spot 3: " + e);
            }

            signalStrengthTextView.setText(String.valueOf(cellSig));
            signalStrengthTextView2.setText(String.valueOf(cellSig2));
            cellIDTextView.setText(String.valueOf(cellID));
            cellMccTextView.setText(String.valueOf(cellMcc));
            cellMncTextView.setText(String.valueOf(cellMnc));
            cellPciTextView.setText(String.valueOf(cellPci));
            cellTacTextView.setText(String.valueOf(cellTac));
            ueDeviceNameTextView.setText(String.valueOf(ueDeviceName));
            ueImeiTextView.setText(String.valueOf(ueIMEI));
            ueImsiTextView.setText(String.valueOf(ueIMSI));

            super.onSignalStrengthsChanged(signalStrength);

            //++++++++++++++++++++++++++++++++++++

        }
    }

    public class LocationService implements LocationListener {

        //The minimum distance to change updates in meters
        private static final long MIN_DISTANCE_CHANGE_FOR_UPDATES = 0; // 10 meters

        //The minimum time between updates in milliseconds
        private static final long MIN_TIME_BW_UPDATES = 0;//1000 * 60 * 1; // 1 minute

        private final static boolean forceNetwork = false;

        private LocationService instance = null;

        private LocationManager locationManager;
        public Location location;
        private boolean isGPSEnabled;
        private boolean isNetworkEnabled;
        private boolean locationServiceAvailable;


        /**
         * Singleton implementation
         * @return
         */
        public LocationService getLocationManager(Context context)     {
            if (instance == null) {
                instance = new LocationService(context);
            }
            return instance;
        }

        /**
         * Local constructor
         */
        private LocationService( Context context )     {

            initLocationService(context);
            Log.d("SignalStrength","LocationService created");
        }



        /**
         * Sets up location service after permissions is granted
         */
        private void initLocationService(Context context) {

            try   {
                longitude = 0.0;
                latitude = 0.0;
                this.locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);

                // Get GPS and network status
                this.isGPSEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
                this.isNetworkEnabled = locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER);

                if (forceNetwork) isGPSEnabled = false;

                if (!isNetworkEnabled && !isGPSEnabled)    {
                    // cannot get location
                    this.locationServiceAvailable = false;
                }
                else
                {
                    this.locationServiceAvailable = true;

                    if (isNetworkEnabled) {
                        locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, MIN_TIME_BW_UPDATES, MIN_DISTANCE_CHANGE_FOR_UPDATES, this);
                        if (locationManager != null)   {
                            location = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                            longitude = this.location.getLongitude();
                            latitude = this.location.getLatitude();
                            updateCoordinates();
                        }
                    }//end if

                    if (isGPSEnabled)  {
                        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, MIN_TIME_BW_UPDATES, MIN_DISTANCE_CHANGE_FOR_UPDATES, this);

                        if (locationManager != null)  {
                            location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
                            longitude = this.location.getLongitude();
                            latitude = this.location.getLatitude();
                            updateCoordinates();
                        }
                    }
                }
            } catch (SecurityException e){
                RequestMultiplePermission();
            } catch (Exception ex)  {
                Log.d("LocationUpdate", "Error creating location service: " + ex.getMessage() );

            }
        }


        @Override
        public void onLocationChanged(Location location)     {
            // do stuff here with location object
            longitude = this.location.getLongitude();
            latitude = this.location.getLatitude();
            updateCoordinates();
        }

        @Override
        public void onStatusChanged(String provider, int status, Bundle extras) {

        }

        @Override
        public void onProviderEnabled(String provider) {

        }

        @Override
        public void onProviderDisabled(String provider) {

        }
    }

}

