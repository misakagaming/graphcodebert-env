namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: MYY10151_PARENT_LIST             Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:42:02
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
  
  public class MYY10151 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10151_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10151_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10151_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY0151_IA Cyyy0151Ia;
    GEN.ORT.YYY.CYYY0151_OA Cyyy0151Oa;
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
    int by___0070254893;
    int limit___0070254893;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020098_esc_flag;
    bool while_0070254893_esc_flag;
    //       +->   MYY10151_PARENT_LIST              01/09/2024  13:42
    //       !       IMPORTS:
    //       !         Work View imp_filter iyy1_list (Transient, Mandatory,
    //       !                     Import only)
    //       !           sort_option
    //       !           scroll_type
    //       !           list_direction
    //       !           scroll_amount
    //       !           order_by_field_num
    //       !         Entity View imp_from iyy1_parent (Transient, Mandatory,
    //       !                     Import only)
    //       !           pinstance_id
    //       !           pkey_attr_text
    //       !         Entity View imp_filter_start iyy1_parent (Transient,
    //       !                     Mandatory, Import only)
    //       !           pkey_attr_text
    //       !         Entity View imp_filter_stop iyy1_parent (Transient,
    //       !                     Mandatory, Import only)
    //       !           pkey_attr_text
    //       !         Entity View imp_filter iyy1_parent (Transient,
    //       !                     Mandatory, Import only)
    //       !           psearch_attr_text
    //       !       EXPORTS:
    //       !         Group View (48) exp_group_list
    //       !           Entity View exp_g_list iyy1_parent (Transient, Export
    //       !                       only)
    //       !             pinstance_id
    //       !             preference_id
    //       !             pkey_attr_text
    //       !             psearch_attr_text
    //       !             pother_attr_text
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
    //       !         Entity View loc_imp_from parent
    //       !           pinstance_id
    //       !           pkey_attr_text
    //       !         Entity View loc_imp_filter parent
    //       !           psearch_attr_text
    //       !         Entity View loc_imp_filter_start parent
    //       !           pkey_attr_text
    //       !         Entity View loc_imp_filter_stop parent
    //       !           pkey_attr_text
    //       !         Group View (48) loc_exp_group_list
    //       !           Entity View loc_exp_g_list parent
    //       !             pinstance_id
    //       !             preference_id
    //       !             pkey_attr_text
    //       !             psearch_attr_text
    //       !             pother_attr_text
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     2 !   
    //     3 !  NOTE: 
    //     3 !  RELEASE HISTORY
    //     3 !  01_00 23-02-1998 New release
    //     4 !   
    //     5 !  NOTE: 
    //     5 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     5 !  !!!!!!!!!!!!
    //     5 !  SET <loc imp*> TO <imp*>
    //     5 !  
    //     6 !  SET loc_imp_filter iyy1_list list_direction TO imp_filter
    //     6 !              iyy1_list list_direction 
    //     7 !  SET loc_imp_filter iyy1_list scroll_type TO imp_filter
    //     7 !              iyy1_list scroll_type 
    //     8 !  SET loc_imp_filter iyy1_list sort_option TO imp_filter
    //     8 !              iyy1_list sort_option 
    //     9 !  SET loc_imp_filter iyy1_list scroll_amount TO imp_filter
    //     9 !              iyy1_list scroll_amount 
    //    10 !  SET loc_imp_filter iyy1_list order_by_field_num TO imp_filter
    //    10 !              iyy1_list order_by_field_num 
    //    11 !   
    //    12 !  SET loc_imp_from parent pinstance_id TO imp_from iyy1_parent
    //    12 !              pinstance_id 
    //    13 !  SET loc_imp_from parent pkey_attr_text TO imp_from iyy1_parent
    //    13 !              pkey_attr_text 
    //    14 !   
    //    15 !  SET loc_imp_filter_start parent pkey_attr_text TO
    //    15 !              imp_filter_start iyy1_parent pkey_attr_text 
    //    16 !  SET loc_imp_filter_stop parent pkey_attr_text TO
    //    16 !              imp_filter_stop iyy1_parent pkey_attr_text 
    //    17 !   
    //    18 !  SET loc_imp_filter parent psearch_attr_text TO imp_filter
    //    18 !              iyy1_parent psearch_attr_text 
    //    19 !   
    //    20 !  NOTE: 
    //    20 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    20 !  !!!!!!!!!!!!
    //    20 !  USE <implementation ab>
    //    20 !  
    //    21 !  USE cyyy0151_parent_list
    //    21 !     WHICH IMPORTS: Work View loc_imp_filter iyy1_list TO Work
    //    21 !              View imp_filter iyy1_list
    //    21 !                    Entity View loc_imp_from parent TO Entity
    //    21 !              View imp_from parent
    //    21 !                    Entity View loc_imp_filter_start parent TO
    //    21 !              Entity View imp_filter_start parent
    //    21 !                    Entity View loc_imp_filter_stop parent TO
    //    21 !              Entity View imp_filter_stop parent
    //    21 !                    Entity View loc_imp_filter parent TO Entity
    //    21 !              View imp_filter parent
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
    //    26 !  !  SET exp_g_list iyy1_parent pinstance_id TO loc_exp_g_list
    //    26 !  !              parent pinstance_id 
    //    27 !  !  SET exp_g_list iyy1_parent preference_id TO loc_exp_g_list
    //    27 !  !              parent preference_id 
    //    28 !  !  SET exp_g_list iyy1_parent pkey_attr_text TO loc_exp_g_list
    //    28 !  !              parent pkey_attr_text 
    //    29 !  !  SET exp_g_list iyy1_parent psearch_attr_text TO
    //    29 !  !              loc_exp_g_list parent psearch_attr_text 
    //    30 !  !  SET exp_g_list iyy1_parent pother_attr_text TO
    //    30 !  !              loc_exp_g_list parent pother_attr_text 
    //    24 !  +--
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public MYY10151(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:42:02";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "MYY10151_PARENT_LIST";
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
    	MYY10151_IA import_view, 
    	MYY10151_OA export_view )
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
      
      f_22020098_localAlloc( "MYY10151_PARENT_LIST" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020098_init(  );
        f_22020098(  );
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
    public void f_22020098(  )
    {
      func_0022020098_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020098" );
      Globdata.GetStateData().SetCurrentABName( "MYY10151_PARENT_LIST" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    See the description for the purpose                             
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <loc imp*> TO <imp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
      WLa.LocImpFilterIyy1ListListDirection = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListListDirection, 1);
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WLa.LocImpFilterIyy1ListScrollType = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListScrollType, 1);
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocImpFilterIyy1ListSortOption = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ListSortOption, 3);
      Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
      WLa.LocImpFilterIyy1ListScrollAmount = IntAttr.ValueOf((int)TIRD2DEC.Execute1((double) WIa.ImpFilterIyy1ListScrollAmount, 0, 
        TIRD2DEC.ROUND_NONE, 5));
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WLa.LocImpFilterIyy1ListOrderByFieldNum = ShortAttr.ValueOf((short)TIRD2DEC.Execute1((double) 
        WIa.ImpFilterIyy1ListOrderByFieldNum, 0, TIRD2DEC.ROUND_NONE, 1));
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      WLa.LocImpFromParentPinstanceId = TimestampAttr.ValueOf(WIa.ImpFromIyy1ParentPinstanceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000013" );
      WLa.LocImpFromParentPkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFromIyy1ParentPkeyAttrText, 5);
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
      WLa.LocImpFilterStartParentPkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterStartIyy1ParentPkeyAttrText, 5);
      Globdata.GetStateData().SetLastStatementNumber( "0000000016" );
      WLa.LocImpFilterStopParentPkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterStopIyy1ParentPkeyAttrText, 5);
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000018" );
      WLa.LocImpFilterParentPsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpFilterIyy1ParentPsearchAttrText, 25);
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <implementation ab>                                         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
      
      Cyyy0151Ia = (GEN.ORT.YYY.CYYY0151_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0151).Assembly,
      	"GEN.ORT.YYY.CYYY0151_IA" ));
      Cyyy0151Oa = (GEN.ORT.YYY.CYYY0151_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0151).Assembly,
      	"GEN.ORT.YYY.CYYY0151_OA" ));
      Cyyy0151Ia.ImpFilterParentPsearchAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterParentPsearchAttrText, 25);
      Cyyy0151Ia.ImpFilterStopParentPkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterStopParentPkeyAttrText, 5);
      Cyyy0151Ia.ImpFilterStartParentPkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpFilterStartParentPkeyAttrText, 5);
      Cyyy0151Ia.ImpFromParentPinstanceId = TimestampAttr.ValueOf(WLa.LocImpFromParentPinstanceId);
      Cyyy0151Ia.ImpFromParentPkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpFromParentPkeyAttrText, 5);
      Cyyy0151Ia.ImpFilterIyy1ListSortOption = FixedStringAttr.ValueOf(WLa.LocImpFilterIyy1ListSortOption, 3);
      Cyyy0151Ia.ImpFilterIyy1ListScrollType = FixedStringAttr.ValueOf(WLa.LocImpFilterIyy1ListScrollType, 1);
      Cyyy0151Ia.ImpFilterIyy1ListListDirection = FixedStringAttr.ValueOf(WLa.LocImpFilterIyy1ListListDirection, 1);
      Cyyy0151Ia.ImpFilterIyy1ListScrollAmount = IntAttr.ValueOf(WLa.LocImpFilterIyy1ListScrollAmount);
      Cyyy0151Ia.ImpFilterIyy1ListOrderByFieldNum = ShortAttr.ValueOf(WLa.LocImpFilterIyy1ListOrderByFieldNum);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY0151).Assembly,
      	"GEN.ORT.YYY.CYYY0151",
      	"Execute",
      	Cyyy0151Ia,
      	Cyyy0151Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020098" );
      Globdata.GetStateData().SetCurrentABName( "MYY10151_PARENT_LIST" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000021" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy0151Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocExpGroupList_MA = IntAttr.ValueOf(Cyyy0151Oa.ExpGroupList_MA);
      for(Adim1 = 1; Adim1 <= WLa.LocExpGroupList_MA; ++(Adim1))
      {
        WLa.LocExpGListParentPinstanceId[Adim1-1] = TimestampAttr.ValueOf(Cyyy0151Oa.ExpGListParentPinstanceId[Adim1-1]);
        WLa.LocExpGListParentPreferenceId[Adim1-1] = TimestampAttr.ValueOf(Cyyy0151Oa.ExpGListParentPreferenceId[Adim1-1]);
        WLa.LocExpGListParentPkeyAttrText[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0151Oa.ExpGListParentPkeyAttrText[Adim1-1], 5);
        WLa.LocExpGListParentPsearchAttrText[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0151Oa.ExpGListParentPsearchAttrText[Adim1-1], 
          25);
        WLa.LocExpGListParentPotherAttrText[Adim1-1] = FixedStringAttr.ValueOf(Cyyy0151Oa.ExpGListParentPotherAttrText[Adim1-1], 25)
          ;
      }
      Cyyy0151Ia.FreeInstance(  );
      Cyyy0151Ia = null;
      Cyyy0151Oa.FreeInstance(  );
      Cyyy0151Oa = null;
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <exp*> TO <loc exp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
      while_0070254893_esc_flag = false;
      LocExpGroupList_PS_002 = (int)1;
      limit___0070254893 = WLa.LocExpGroupList_MA;
      by___0070254893 = 1;
      while ((LocExpGroupList_PS_002 <= limit___0070254893) && (while_0070254893_esc_flag != true))
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
          WOa.ExpGListIyy1ParentPinstanceId[ExpGroupList_PS_001-1] = TimestampAttr.ValueOf(WLa.LocExpGListParentPinstanceId[
            LocExpGroupList_PS_002-1]);
          f_173015047_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000027" );
          WOa.ExpGListIyy1ParentPreferenceId[ExpGroupList_PS_001-1] = TimestampAttr.ValueOf(WLa.LocExpGListParentPreferenceId[
            LocExpGroupList_PS_002-1]);
          f_173015047_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000028" );
          WOa.ExpGListIyy1ParentPkeyAttrText[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListParentPkeyAttrText[
            LocExpGroupList_PS_002-1], 5);
          f_173015047_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000029" );
          WOa.ExpGListIyy1ParentPsearchAttrText[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(
            WLa.LocExpGListParentPsearchAttrText[LocExpGroupList_PS_002-1], 25);
          f_173015047_rgvc(  );
          Globdata.GetStateData().SetLastStatementNumber( "0000000030" );
          WOa.ExpGListIyy1ParentPotherAttrText[ExpGroupList_PS_001-1] = FixedStringAttr.ValueOf(WLa.LocExpGListParentPotherAttrText[
            LocExpGroupList_PS_002-1], 25);
          f_173015047_rgvc(  );
        }
        Globdata.GetStateData().SetLastStatementNumber( "0000000024" );
        LocExpGroupList_PS_002 = (int)(LocExpGroupList_PS_002 + by___0070254893);
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
    
    public void f_22020098_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.MYY10151_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.MYY10151_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020098" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020098_init(  )
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
        WOa.ExpGListIyy1ParentPinstanceId[ExpGroupList_PS_001-1] = "00000000000000000000";
        WOa.ExpGListIyy1ParentPreferenceId[ExpGroupList_PS_001-1] = "00000000000000000000";
        WOa.ExpGListIyy1ParentPkeyAttrText[ExpGroupList_PS_001-1] = "     ";
        WOa.ExpGListIyy1ParentPsearchAttrText[ExpGroupList_PS_001-1] = "                         ";
        WOa.ExpGListIyy1ParentPotherAttrText[ExpGroupList_PS_001-1] = "                         ";
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
    public void f_173015047_rgvc(  )
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
