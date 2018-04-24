package dk.dtu.mobileai.modules;

import android.content.Context;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.util.Log;

import java.text.DecimalFormat;

import dk.dtu.mobileai.data.DataStore;
import dk.dtu.mobileai.data.IApiEndpoint;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class LocationModule implements LocationListener {

    private static final String TAG = "LocationModule";

    private Context mContext;
    private LocationManager mLocationManager;
    //The minimum distance to change updates in meters
    private static final long MIN_DISTANCE_CHANGE_FOR_UPDATES = 0; // 10 meters

    //The minimum time between updates in milliseconds
    //private static final long MIN_TIME_BW_UPDATES = 1000 * 60 * 1; // 1 minute

    private static final long MIN_TIME_BW_UPDATES = 0; // 1 second

    private final static boolean forceNetwork = false;

    public Location location;
    private boolean isGPSEnabled;
    private boolean isNetworkEnabled;
    private boolean locationServiceAvailable;

    private double longitude;
    private double latitude;


    /**
     * Local constructor
     */
    public LocationModule(Context context)     {
        mContext = context;
        Log.d(TAG,"LocationService created");

        mLocationManager = (LocationManager) mContext.getSystemService(Context.LOCATION_SERVICE);
        initLocationService();
    }



    /**
     * Sets up location service after permissions is granted
     */
    private void initLocationService() {

        try   {
            longitude = 0.0;
            latitude = 0.0;
            this.mLocationManager = (LocationManager) mContext.getSystemService(Context.LOCATION_SERVICE);

            // Get GPS and network status
            this.isGPSEnabled = mLocationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
            this.isNetworkEnabled = mLocationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER);

            if (forceNetwork) isGPSEnabled = false;

            if (!isNetworkEnabled && !isGPSEnabled)    {
                // cannot get location
                this.locationServiceAvailable = false;
            }
            //else
            {
                this.locationServiceAvailable = true;

                if (isNetworkEnabled) {
                    mLocationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, MIN_TIME_BW_UPDATES, MIN_DISTANCE_CHANGE_FOR_UPDATES, this);
                    if (mLocationManager != null)   {
                        location = mLocationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
                        longitude = this.location.getLongitude();
                        latitude = this.location.getLatitude();
                        updateCoordinates();
                    }
                }//end if

                if (isGPSEnabled)  {
                    mLocationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, MIN_TIME_BW_UPDATES, MIN_DISTANCE_CHANGE_FOR_UPDATES, this);

                    if (mLocationManager != null)  {
                        location = mLocationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
                        longitude = this.location.getLongitude();
                        latitude = this.location.getLatitude();
                        updateCoordinates();
                    }
                }
            }
        } catch (SecurityException e){
            Log.e("LocationUpdate", e.getMessage() );
        } catch (Exception ex)  {
            Log.e("LocationUpdate", ex.getMessage() );

        }
    }


    @Override
    public void onLocationChanged(Location location)     {
        // do stuff here with location object
        this.location = location;
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

    public void updateCoordinates(){
        Log.d("LocationUpdate", "latitude: " + latitude + " longitude: " + longitude);

        DataStore dataStore = DataStore.getInstance();

        dataStore.notifyDataChanged();

        if (dataStore.apiReady()){
            IApiEndpoint apiService = dataStore.retrofit.create(IApiEndpoint.class);

            dk.dtu.mobileai.models.Location loc = new dk.dtu.mobileai.models.Location();
            loc.setEntity(dataStore.getApiEndpoint() + "entities/" + dataStore.getApiEntityId() + "/");

            DecimalFormat df = new DecimalFormat("#.######");

            loc.setLatitude(Double.valueOf(df.format(latitude)));
            loc.setLongitude(Double.valueOf(df.format(longitude)));

            Call<dk.dtu.mobileai.models.Location> call = apiService.createLocation(DataStore.getInstance().getApiEntityId(), loc);

            try {
                call.enqueue(new Callback<dk.dtu.mobileai.models.Location>() {
                    @Override
                    public void onResponse(Call<dk.dtu.mobileai.models.Location> call, Response<dk.dtu.mobileai.models.Location> response) {
                        Log.d(TAG, response.toString());
                    }

                    @Override
                    public void onFailure(Call<dk.dtu.mobileai.models.Location> call, Throwable t) {
                        Log.d(TAG, t.toString());
                    }
                });
            } catch (Exception e ){
                Log.e(TAG, e.getMessage());
            }
        }

    }
}
