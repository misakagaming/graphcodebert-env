namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: MYY10351_TYPE_LIST               Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:36
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class MYY10351 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10351_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10351_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10351_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY0351_IA Cyyy0351Ia;
    GEN.ORT.YYY.CYYY0351_OA Cyyy0351Oa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // REPEATING GROUP VIEW STATUS FIELDS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool ExpGroupList_FL_001;
    int ExpGroupList_PS_001;
    bool ExpGroupList_RF_001;
    public const int ExpGroupList_MM_001 = 48;
    bool LocExpGroupList_FL_002;
    int LocExpGroupList_PS_002;
    bool LocExpGroupList_RF_002;
    public const int LocExpGroupList_MM_002 = 48;
    bool ExpGroupList_FL_003;
    int ExpGroupList_PS_003;
    bool ExpGroupList_RF_003;
    public const int ExpGroupList_MM_003 = 48;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // FOR-LOOP CONTROL TEMPORARY VARIABLES 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    int by___0070254835;
    int limit___0070254835;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020161_esc_flag;
    bool while_0070254835_esc_flag;
    //       +->   MYY10351_TYPE_LIST                01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_filter iyy1_list (Transient, Mandatory,
    //       !                     Import only)
    //       !           sort_option
    //       !           scroll_type
    //       !           list_direction
    //       !           scroll_amount
    //       !           order_by_field_num
    //       !         Entity View imp_from iyy1_type (Transient, Mandatory,
    //       !                     Import only)
    //       !           tinstance_id
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !         Entity View imp_filter_start iyy1_type (Transient,
    //       !                     Mandatory, Import only)
    //       !           tkey_attr_text
    //       !         Entity View imp_filter_stop iyy1_type (Transient,
    //       !                     Mandatory, Import only)
    //       !           tkey_attr_text
    //       !         Entity View imp_filter iyy1_type (Transient, Mandatory,
    //       !                     Import only)
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !       EXPORTS:
    //       !         Group View (48) exp_group_list
    //       !           Entity View exp_g_list iyy1_type (Transient, Export
    //       !                       only)
    //       !             tinstance_id
    //       !             treference_id
    //       !             tcreate_user_id
    //       !             tupdate_user_id
    //       !             tkey_attr_text
    //       !             tsearch_attr_text
    //       !             tother_attr_text
    //       !             tother_attr_date
    //       !             tother_attr_time
    //       !             tother_attr_amount
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Work View loc_imp_filter iyy1_list
    //       !           sort_option
    //       !           scroll_type
    //       !           list_direction
    //       !           scroll_amount
    //       !           order_by_field_num
    //       !         Entity View loc_imp_from type
    //       !           tinstance_id
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !         Entity View loc_imp_filter_start type
    //       !           tkey_attr_text
    //       !         Entity View loc_imp_filter_stop type
    //       !           tkey_attr_text
    //       !         Entity View loc_imp_filter type
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !         Group View (48) loc_exp_group_list
    //       !           Entity View loc_exp_g_list type
    //       !             tinstance_id
    //       !             treference_id
    //       !             tcreate_user_id
    //       !             tupdate_user_id
    //       !             tkey_attr_text
    //       !             tsearch_attr_text
    //       !             tother_attr_text
    //       !             tother_attr_date
    //       !             tother_attr_time
    //       !             tother_attr_amount
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  RELEASE HISTORY
    //     2 !  01_00 23-02-1998 New release
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     3 !  !!!!!!!!!!!!
    //     3 !  SET <loc imp*> TO <imp*>
    //     3 !  
    //     4 !  SET loc_imp_filter iyy1_list list_direction TO imp_filter
    //     4 !              iyy1_list list_direction 
    //     5 !  SET loc_imp_filter iyy1_list scroll_type TO imp_filter
    //     5 !              iyy1_list scroll_type 
    //     6 !  SET loc_imp_filter iyy1_list sort_option TO imp_filter
    //     6 !              iyy1_list sort_option 
    //     7 !  SET loc_imp_filter iyy1_list scroll_amount TO imp_filter
    //     7 !              iyy1_list scroll_amount 
    //     8 !  SET loc_imp_filter iyy1_list order_by_field_num TO imp_filter
    //     8 !              iyy1_list order_by_field_num 
    //     9 !   
    //    10 !  SET loc_imp_from type tinstance_id TO imp_from iyy1_type
    //    10 !              tinstance_id 
    //    11 !  SET loc_imp_from type tkey_attr_text TO imp_from iyy1_type
    //    11 !              tkey_attr_text 
    //    12 !  SET loc_imp_from type tsearch_attr_text TO imp_from iyy1_type
    //    12 !              tsearch_attr_text 
    //    13 !   
    //    14 !  SET loc_imp_filter_start type tkey_attr_text TO
    //    14 !              imp_filter_start iyy1_type tkey_attr_text 
    //    15 !  SET loc_imp_filter_stop type tkey_attr_text TO imp_filter_stop
    //    15 !              iyy1_type tkey_attr_text 
    //    16 !   
    //    17 !  SET loc_imp_filter type tsearch_attr_text TO imp_filter
    //    17 !              iyy1_type tsearch_attr_text 
    //    18 !  SET loc_imp_filter type tother_attr_text TO imp_filter
    //    18 !              iyy1_type tother_attr_text 
    //    19 !   
    //    20 !  NOTE: 
    //    20 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    20 !  !!!!!!!!!!!!
    //    20 !  USE <implementation ab>
    //    20 !  
    //    21 !  USE cyyy0351_type_list
    //    21 !     WHICH IMPORTS: Work View loc_imp_filter iyy1_list TO Work
    //    21 !              View imp_filter iyy1_list
    //    21 !                    Entity View loc_imp_from type TO Entity View
    //    21 !              imp_from type
    //    21 !                    Entity View loc_imp_filter_start type TO
    //    21 !              Entity View imp_filter_start type
    //    21 !                    Entity View loc_imp_filter_stop type TO
    //    21 !              Entity View imp_filter_stop type
    //    21 !                    Entity View loc_imp_filter type TO Entity
    //    21 !              View imp_filter type
    //    21 !     WHICH EXPORTS: Group View  loc_exp_group_list FROM Group
    //    21 !              View exp_group_list
    //    21 !                    Work View exp_error iyy1_component FROM Work
    //    21 !              View exp_error iyy1_component
    //    22 !   
    //    23 !  NOTE: 
    //    23 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    23 !  !!!!!!!!!!!!
    //    23 !  SET <exp*> TO <loc exp*>
    //    23 !  
    //    24 !  +=>FOR SUBSCRIPT OF loc_exp_group_list FROM 1 TO LAST OF
    //    24 !  !        loc_exp_group_list BY 1
    //    25 !  !  SET SUBSCRIPT OF exp_group_list TO SUBSCRIPT OF
    //    25 !  !              loc_exp_group_list 
    //    26 !  !  SET exp_g_list iyy1_type tinstance_id TO loc_exp_g_list
    //    26 !  !              type tinstance_id 
    //    27 !  !  SET exp_g_list iyy1_type treference_id TO loc_exp_g_list
    //    27 !  !              type treference_id 
    //    28 !  !  SET exp_g_list iyy1_type tcreate_user_id TO loc_exp_g_list
    //    28 !  !              type tcreate_user_id 
    //    29 !  !  SET exp_g_list iyy1_type tupdate_user_id TO loc_exp_g_list
    //    29 !  !              type tupdate_user_id 
    //    30 !  !  SET exp_g_list iyy1_type tkey_attr_text TO loc_exp_g_list
    //    30 !  !              type tkey_attr_text 
    //    31 !  !  SET exp_g_list iyy1_type tsearch_attr_text TO
    //    31 !  !              loc_exp_g_list type tsearch_attr_text 
    //    32 !  !  SET exp_g_list iyy1_type tother_attr_text TO loc_exp_g_list
    //    32 !  !              type tother_attr_text 
    //    33 !  !  SET exp_g_list iyy1_type tother_attr_date TO loc_exp_g_list
    //    33 !  !              type tother_attr_date 
    //    34 !  !  SET exp_g_list iyy1_type tother_attr_time TO loc_exp_g_list
    //    34 !  !              type tother_attr_time 
    //    35 !  !  SET exp_g_list iyy1_type tother_attr_amount TO
    //    35 !  !              loc_exp_g_list type tother_attr_amount 
    //    24 !  +--
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public MYY10351(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:36";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "MYY10351_TYPE_LIST";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	MYY10351_IA import_view, 
    	MYY10351_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020161_localAlloc( "MYY10351_TYPE_LIST" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020161_init(  );
        f_22020161(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020161(  )
    {
      func_0022020161_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020161" );
      Globdata.GetStateData().SetCurrentABName( "MYY10351_TYPE_LIST" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    See the description for the purpose                             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <loc imp*> TO <imp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000004" );
      WLa.LocImpFilterIyy1ListListDirection = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListListDirection, 1);
      Globdata.GetStateData().SetLastStatementNumber( "0000000005" );
      WLa.LocImpFilterIyy1ListScrollType = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListScrollType, 1);
      Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
      WLa.LocImpFilterIyy1ListSortOption = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListSortOption, 3);
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WLa.LocImpFilterIyy1ListScrollAmount = IntAttr.ValueOf((int)TIRD2DEC.Execute1((double) WIa.ImpFilterIyy1ListScrollAmount, 0, 
        TIRD2DEC.ROUND_NONE, 5));
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocImpFilterIyy1ListOrderByFieldNum = ShortAttr.ValueOf((short)TIRD2DEC.Execute1((double) 
        WIa.ImpFilterIyy1ListOrderByFieldNum, 0, TIRD2DEC.ROUND_NONE, 1));
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WLa.LocImpFromTypeTinstanceId = TimestampAttr.ValueOf(WIa.ImpFromIyy1TypeTinstanceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000011" );
      WLa.LocImpFromTypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFromIyy1TypeTkeyAttrText, 4);
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      WLa.LocImpFromTypeTsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpFromIyy1TypeTsearchAttrText, 20);
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000014" );
      WLa.LocImpFilterStartTypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterStartIyy1TypeTkeyAttrText, 4);
      Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
      WLa.LocImpFilterStopTypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterStopIyy1TypeTkeyAttrText, 4);
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000017" );
      WLa.LocImpFilterTypeTsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1TypeTsearchAttrText, 20);
      Globdata.GetStateData().SetLastStatementNumber( "0000000018" );
      WLa.LocImpFilterTypeTotherAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1TypeTotherAttrText, 2);
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <implementation ab>                                         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
      
      Cyyy0351Ia = (GEN.ORT.YYY.CYYY0351_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0351).Assembly,
      	"GEN.ORT.YYY.CYYY0351_IA" ));
      Cyyy0351Oa = (GEN.ORT.YYY.CYYY0351_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0351).Assembly,
      	"GEN.ORT.YYY.CYYY0351_OA" ));
      Cyyy0351Ia.ImpFilterTypeTsearchAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterTypeTsearchAttrText, 20);
      Cyyy0351Ia.ImpFilterTypeTotherAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterTypeTotherAttrText, 2);
      Cyyy0351Ia.ImpFilterStopTypeTkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterStopTypeTkeyAttrText, 4);
      Cyyy0351Ia.ImpFilterStartTypeTkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterStartTypeTkeyAttrText, 4);
      Cyyy0351Ia.ImpFromTypeTinstanceId = TimestampAttr.ValueOf(WLa.LocImpFromTypeTinstanceId);
      Cyyy0351Ia.ImpFromTypeTkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpFromTypeTkeyAttrText, 4);
      Cyyy0351Ia.ImpFromTypeTsearchAttrText = FixedStringAttr.ValueOf(WLa.LocImpFromTypeTsearchAttrText, 20);
      Cyyy0351Ia.ImpFilterIyy1ListSortOption = FixedStringAttr.ValueOf(WLa.LocImpFilterIyy1ListSortOption, 3);
      Cyyy0351Ia.ImpFilterIyy1ListScrollType = FixedStringAttr.ValueOf(WLa.LocImpFilterIyy1ListScrollType, 1);
      Cyyy0351Ia.ImpFilterIyy1ListListDirection = FixedStringAttr.ValueOf(WLa.LocImpFilterIyy1ListListDirection, 1);
      Cyyy0351Ia.ImpFilterIyy1ListScrollAmount = IntAttr.ValueOf(WLa.LocImpFilterIyy1ListScrollAmount);
      Cyyy0351Ia.ImpFilterIyy1ListOrderByFieldNum = ShortAttr.ValueOf(WLa.LocImpFilterIyy1ListOrderByFieldNum);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY0351).Assembly,
      	"GEN.ORT.YYY.CYYY0351",
      	"Execute",
      	Cyyy0351Ia,
      	Cyyy0351Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020161" );
      Globdata.GetStateData().SetCurrentABName( "MYY10351_TYPE_LIST" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocExpGroupList_MA = IntAttr.ValueOf(Cyyy0351Oa.ExpGroupList_MA);
      for(Adim1 = 1; Adim1 <= WLa.LocExpGroupList_MA; ++(Adim1))
      {
        WLa.LocExpGListTypeTinstanceId[Adim1-1] = TimestampAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTinstanceId[Adim1-1]);
        WLa.LocExpGListTypeTreferenceId[Adim1-1] = TimestampAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTreferenceId[Adim1-1]);
        WLa.LocExpGListTypeTcreateUserId[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTcreateUserId[Adim1-1], 8);
        WLa.LocExpGListTypeTupdateUserId[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTupdateUserId[Adim1-1], 8);
        WLa.LocExpGListTypeTkeyAttrText[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTkeyAttrText[Adim1-1], 4);
        WLa.LocExpGListTypeTsearchAttrText[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTsearchAttrText[Adim1-1], 20);
        WLa.LocExpGListTypeTotherAttrText[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTotherAttrText[Adim1-1], 2);
        WLa.LocExpGListTypeTotherAttrDate[Adim1-1] = DateAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTotherAttrDate[Adim1-1]);
        WLa.LocExpGListTypeTotherAttrTime[Adim1-1] = TimeAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTotherAttrTime[Adim1-1]);
        WLa.LocExpGListTypeTotherAttrAmount[Adim1-1] = DecimalAttr.ValueOf(Cyyy0351Oa.ExpGListTypeTotherAttrAmount[Adim1-1]);
      }
      Cyyy0351Ia.FreeInstance(  );
      Cyyy0351Ia = null;
      Cyyy0351Oa.FreeInstance(  );
      Cyyy0351Oa = null;
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <exp*> TO <loc exp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
      while_0070254835_esc_flag = false;
      LocExpGroupList_PS_002 = (int)1;
      limit___0070254835 = WLa.LocExpGroupList_MA;
      by___0070254835 = 1;
      while ((LocExpGroupList_PS_002 <= limit___0070254835) && (while_0070254835_esc_flag != true))
      {
        if ( (LocExpGroupList_PS_002 > 48) || (LocExpGroupList_PS_002 < 1) )
        break;
        if ( (LocExpGroupList_PS_002 > WLa.LocExpGroupList_MA) && (LocExpGroupList_PS_002 <= 48) )
        WLa.LocExpGroupList_MA = IntAttr.ValueOf(LocExpGroupList_PS_002);
        {
          Globdata.GetStateData().SetLastStatementNumber( "0000000025" );
          ExpGroupList_PS_001 = (int)TIRD2DEC.Execute1(LocExpGroupList_PS_002, 0, TIRD2DEC.ROUND_NONE, 0);
          if ( (ExpGroupList_PS_001 > WOa.ExpGroupList_MA) && (ExpGroupList_PS_001 <= 48) )
          WOa.ExpGroupList_MA = IntAttr.ValueOf(ExpGroupList_PS_001);
          Globdata.GetStateData().SetLastStatementNumber( "0000000026" );
          WOa.ExpGListIyy1TypeTinstanceId[ExpGroupList_PS_001-1] = TimestampAttr.ValueOf(WLa.LocExpGListTypeTinstanceId[
            LocExpGroupList_PS_002-1]);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000027" );
          WOa.ExpGListIyy1TypeTreferenceId[ExpGroupList_PS_001-1] = TimestampAttr.ValueOf(WLa.LocExpGListTypeTreferenceId[
            LocExpGroupList_PS_002-1]);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000028" );
          WOa.ExpGListIyy1TypeTcreateUserId[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListTypeTcreateUserId[
            LocExpGroupList_PS_002-1], 8);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000029" );
          WOa.ExpGListIyy1TypeTupdateUserId[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListTypeTupdateUserId[
            LocExpGroupList_PS_002-1], 8);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000030" );
          WOa.ExpGListIyy1TypeTkeyAttrText[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListTypeTkeyAttrText[
            LocExpGroupList_PS_002-1], 4);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000031" );
          WOa.ExpGListIyy1TypeTsearchAttrText[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListTypeTsearchAttrText[
            LocExpGroupList_PS_002-1], 20);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000032" );
          WOa.ExpGListIyy1TypeTotherAttrText[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListTypeTotherAttrText[
            LocExpGroupList_PS_002-1], 2);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000033" );
          WOa.ExpGListIyy1TypeTotherAttrDate[ExpGroupList_PS_001-1] = DateAttr.ValueOf(WLa.LocExpGListTypeTotherAttrDate[
            LocExpGroupList_PS_002-1]);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000034" );
          WOa.ExpGListIyy1TypeTotherAttrTime[ExpGroupList_PS_001-1] = TimeAttr.ValueOf(WLa.LocExpGListTypeTotherAttrTime[
            LocExpGroupList_PS_002-1]);
          f_173015127_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000035" );
          WOa.ExpGListIyy1TypeTotherAttrAmount[ExpGroupList_PS_001-1] = DecimalAttr.ValueOf(TIRBDTRU.TruncateToDecimal( 
            WLa.LocExpGListTypeTotherAttrAmount[LocExpGroupList_PS_002-1], 2));
          f_173015127_rgvc(  );
        }
        Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
        LocExpGroupList_PS_002 = (int)(LocExpGroupList_PS_002 + by___0070254835);
        Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
      }
      if ( LocExpGroupList_PS_002 > 48 )
      LocExpGroupList_PS_002 = 48;
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020161_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.MYY10351_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.MYY10351_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020161" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020161_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WLa.LocExpGroupList_MA = 0;
      for(LocExpGroupList_PS_002 = 1; LocExpGroupList_PS_002 <= 48; ++(LocExpGroupList_PS_002))
      {
      }
      WOa.ExpGroupList_MA = 0;
      for(ExpGroupList_PS_001 = 1; ExpGroupList_PS_001 <= 48; ++(ExpGroupList_PS_001))
      {
        WOa.ExpGListIyy1TypeTinstanceId[ExpGroupList_PS_001-1] = "00000000000000000000";
        WOa.ExpGListIyy1TypeTreferenceId[ExpGroupList_PS_001-1] = "00000000000000000000";
        WOa.ExpGListIyy1TypeTcreateUserId[ExpGroupList_PS_001-1] = "        ";
        WOa.ExpGListIyy1TypeTupdateUserId[ExpGroupList_PS_001-1] = "        ";
        WOa.ExpGListIyy1TypeTkeyAttrText[ExpGroupList_PS_001-1] = "    ";
        WOa.ExpGListIyy1TypeTsearchAttrText[ExpGroupList_PS_001-1] = "                    ";
        WOa.ExpGListIyy1TypeTotherAttrText[ExpGroupList_PS_001-1] = "  ";
        WOa.ExpGListIyy1TypeTotherAttrDate[ExpGroupList_PS_001-1] = 00000000;
        WOa.ExpGListIyy1TypeTotherAttrTime[ExpGroupList_PS_001-1] = 00000000;
        WOa.ExpGListIyy1TypeTotherAttrAmount[ExpGroupList_PS_001-1] = DecimalAttr.GetDefaultValue();
      }
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
      ExpGroupList_PS_001 = 1;
      LocExpGroupList_PS_002 = 1;
    }
    public void f_173015127_rgvc(  )
    {
      if ( (ExpGroupList_PS_001 > 48) || (ExpGroupList_PS_001 < 1) )
      {
        Globdata.GetErrorData().SetViewOverflow( ErrorData.ErrorEncounteredErrorFound );
        {
          throw new ABException();
        }
      }
    }
  }// end class
  
}

