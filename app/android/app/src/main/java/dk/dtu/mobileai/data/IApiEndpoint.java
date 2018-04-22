package dk.dtu.mobileai.data;

import java.util.List;

import dk.dtu.mobileai.models.Entity;
import dk.dtu.mobileai.models.Location;
import dk.dtu.mobileai.models.Measurement;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Headers;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface IApiEndpoint {
    // Request method and URL specified in the annotation

    @GET("entities/{id}")
    Call<Entity> getEntity(@Path("entities") String id);

    @GET("entities/{id}/")
    Call<List<Entity>> getEntities(@Path("id") String id);

    @POST("entities/")
    Call<Entity> createEntity(@Body Entity entity);

    @POST("entities/{id}/locations/")
    Call<Location> createLocation(@Path("id") String id, @Body Location location);

    @POST("entities/{id}/measurements/")
    Call<Measurement> createMeasurement(@Path("id") String id, @Body Measurement measurement);
}