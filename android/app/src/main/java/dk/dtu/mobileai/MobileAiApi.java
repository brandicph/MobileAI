package dk.dtu.mobileai;

import java.sql.Timestamp;
import java.util.Date;
import java.util.List;

import retrofit2.Call;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface MobileAiApi {

    @GET("cellinfolte/")
    Call<List<CellInfoLte>> getAllCellInfoLte();

    @GET("cellinfolte/{pk}/")
    Call<CellInfoLte> getCellInfoLte(
            @Path("pk") Integer pk);

    @POST("cellinfolte/")
    @FormUrlEncoded
    Call<CellInfoLte> postCellInfoLte(
            @Field("registered") boolean registered,
            @Field("timestamp") String timestamp);

    @POST("cellidentitylte/")
    @FormUrlEncoded
    Call<CellIdentityLte> postCellIdentityLte(
            @Field("mcc") long mcc,
            @Field("tac") long tac,
            @Field("earfcn") long earfcn,
            @Field("ci") long ci,
            @Field("mnc") long mnc,
            @Field("pci") long pci,
            @Field("cell_info_lte") String cell_info_lte);

    @POST("cellsignalstrengthlte/")
    @FormUrlEncoded
    Call<CellSignalStrengthLte> postCellSignalStrengthLte(
            @Field("cqi") long cqi,
            @Field("rssnr") long rssnr,
            @Field("ta") long ta,
            @Field("ss") long ss,
            @Field("rsrp") long rsrp,
            @Field("rsrq") long rsrq,
            @Field("cell_info_lte") String cell_info_lte);

    public static final Retrofit retrofit = new Retrofit.Builder()
            .baseUrl("http://192.168.1.4:8000/")
            .addConverterFactory(GsonConverterFactory.create())
            .build();
}
